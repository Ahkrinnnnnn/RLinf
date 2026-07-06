# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin TD3 + BC training driver that reuses lerobot's shared RLT core.

This is *not* a Ray worker. It is a plain Python object that wires the shared
``td3_critic_loss`` / ``rlt_actor_loss`` / ``polyak_update`` math from
``lerobot.rlt.shared`` to optimizers, so RLinf's training backend can drive the
exact same algorithm as lerobot's ``RLTTD3Algorithm``. The two are numerically
identical by construction (same code path) — verified by the conformance test.

The intended RLinf integration is: a TrainingWorker wraps this driver and calls
``update_step`` per gradient step; a RolloutWorker uses
:class:`~rlinf.rlt.chunked_rollout.ChunkedRolloutRunner` to collect, then ships
``ReplayWindow`` batches to the learner. Actor-weight sync between the two
workers reuses ``get_actor_weights`` / ``load_actor_weights`` (and, in a
distributed setting, RLinf's existing ``WeightSyncer``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.rlt_actor import ChunkActor, RLTActorConfig
from lerobot.rlt.shared import (
    ChunkCriticEnsemble,
    bc_weight_schedule,
    polyak_update,
    rlt_actor_loss,
    td3_critic_loss,
)


@dataclass
class RLTTD3DriverConfig:
    """Hyperparameters mirroring ``RLTTD3AlgorithmConfig`` (subset)."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount: float = 0.99
    critic_target_update_weight: float = 0.005
    num_critics: int = 2
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    utd_ratio: int = 1
    policy_update_freq: int = 2
    grad_clip_norm: float = 40.0
    bc_weight_max: float = 1.0
    bc_weight_min: float = 0.0
    bc_decay_steps: int = 10000
    warmup_pretraining_updates: int = 0
    delta_penalty_weight: float = 0.0
    ref_dropout_prob: float = 0.0
    use_adamw: bool = False
    optimizer_weight_decay: float = 0.01


class RLTTD3Driver:
    """Plain TD3 + BC learner reusing lerobot's shared loss math.

    The driver owns the actor, the critic ensemble (online + target) and the
    optimizers. It does *not* own a frozen VLA / RL-token encoder — at rollout
    time the caller supplies ``z_rl`` and ``ref_chunk`` via the forward batch
    (``fb``), exactly as lerobot's collector does.
    """

    def __init__(
        self,
        actor_config: RLTActorConfig,
        config: RLTTD3DriverConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.actor_config = actor_config
        self.config = config or RLTTD3DriverConfig()
        self.device = torch.device(device)

        self.actor = ChunkActor(actor_config).to(self.device)
        chunk_dim = actor_config.chunk_len * actor_config.action_dim
        hidden = list(self.config.critic_hidden_dims)
        self.critic_ensemble = ChunkCriticEnsemble(
            token_dim=actor_config.token_dim,
            proprio_dim=actor_config.proprio_dim,
            chunk_dim=chunk_dim,
            num_critics=self.config.num_critics,
            hidden_dims=hidden,
        ).to(self.device)
        self.critic_target = ChunkCriticEnsemble(
            token_dim=actor_config.token_dim,
            proprio_dim=actor_config.proprio_dim,
            chunk_dim=chunk_dim,
            num_critics=self.config.num_critics,
            hidden_dims=hidden,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        self._step = 0
        self._make_optimizers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _make_optimizers(self) -> None:
        cls = torch.optim.AdamW if self.config.use_adamw else torch.optim.Adam
        kw = {}
        if self.config.use_adamw:
            kw["weight_decay"] = self.config.optimizer_weight_decay
        self.opt_actor = cls(self.actor.parameters(), lr=self.config.actor_lr, **kw)
        self.opt_critic = cls(self.critic_ensemble.parameters(), lr=self.config.critic_lr, **kw)

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    def _bc_weight(self) -> float:
        return bc_weight_schedule(
            self._step,
            self.config.bc_weight_max,
            self.config.bc_weight_min,
            self.config.bc_decay_steps,
        )

    def _in_warmup(self) -> bool:
        return self._step < self.config.warmup_pretraining_updates

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_step(self, fb: dict[str, Any]) -> dict[str, float]:
        """Run one TD3 outer step (``utd_ratio`` critic updates + delayed actor).

        ``fb`` is the shared forward-batch dict (see ``lerobot.rlt.shared``).
        Returns a stats dict with ``loss_critic`` and (when the actor is
        updated) ``loss_actor`` / ``rl_loss`` / ``bc_loss`` / ``delta_penalty``
        / ``bc_weight``.
        """
        clip = self.config.grad_clip_norm
        stats: dict[str, float] = {}

        for _ in range(self.config.utd_ratio - 1):
            loss_c = td3_critic_loss(
                self.critic_ensemble, self.critic_target, self.actor, fb, self.config.discount
            )
            self.opt_critic.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.opt_critic.step()
            polyak_update(self.critic_target, self.critic_ensemble, self.config.critic_target_update_weight)

        loss_critic = td3_critic_loss(
            self.critic_ensemble, self.critic_target, self.actor, fb, self.config.discount
        )
        self.opt_critic.zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic_ensemble.parameters(), max_norm=clip
        ).item()
        self.opt_critic.step()
        stats["loss_critic"] = float(loss_critic.detach().item())
        stats["critic_grad_norm"] = critic_grad

        if self._step % self.config.policy_update_freq == 0:
            loss_actor, info = rlt_actor_loss(
                self.actor,
                self.critic_ensemble,
                fb,
                chunk_len=self.actor_config.chunk_len,
                action_dim=self.actor_config.action_dim,
                bc_weight=self._bc_weight(),
                delta_penalty_weight=self.config.delta_penalty_weight,
                ref_dropout_prob=self.config.ref_dropout_prob,
                in_warmup=self._in_warmup(),
            )
            self.opt_actor.zero_grad()
            loss_actor.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=clip
            ).item()
            self.opt_actor.step()
            stats["loss_actor"] = float(loss_actor.detach().item())
            stats["actor_grad_norm"] = actor_grad
            stats["rl_loss"] = info["rl_loss"]
            stats["bc_loss"] = info["bc_loss"]
            stats["delta_penalty"] = info["delta_penalty"]
            stats["bc_weight"] = self._bc_weight()

        polyak_update(self.critic_target, self.critic_ensemble, self.config.critic_target_update_weight)
        self._step += 1
        return stats

    # ------------------------------------------------------------------
    # Weight sync (sim ↔ real transfer)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_actor_weights(self) -> dict[str, Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}

    def load_actor_weights(self, weights: dict[str, Tensor]) -> None:
        self.actor.load_state_dict({k: v.to(self.device) for k, v in weights.items()})

    # ------------------------------------------------------------------
    # Inference helper (rollout-side)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_chunk_mean(self, z_rl: Tensor, proprio: Tensor, ref_chunk: Tensor) -> Tensor:
        """Deterministic actor mean ``[B, C*d]`` for open-loop chunk execution."""
        self.actor.eval()
        return self.actor(z_rl.to(self.device), proprio.to(self.device), ref_chunk.to(self.device))[2]
