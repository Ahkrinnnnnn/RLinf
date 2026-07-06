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

"""Chunked open-loop rollout runner reusing lerobot's ``StepRecord`` / replay windows.

RLinf's simulation rollout worker calls this runner to produce replay windows
in the *exact* format lerobot's collector emits, so the learner (RLinf or
lerobot) consumes identical transitions. The runner is environment-agnostic:
the caller supplies a ``machine_a_fn`` (frozen VLA + RL-token query returning
``(z_rl, ref_chunk)``) and an ``env_step_fn`` (execute an action, return
``proprio, reward, done``). For unit tests a :class:`MockChunkEnv` is provided.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from lerobot.rlt.replay_windows import (
    SOURCE_BASE,
    SOURCE_RL,
    StepRecord,
    build_replay_windows,
)


class MockChunkEnv:
    """Deterministic toy environment for conformance / smoke tests.

    State = proprio vector of length ``proprio_dim``. Action = vector of
    length ``action_dim``. The dynamics are a simple linear push plus a
    success condition (reach a target). No physics — purely for exercising the
    chunked rollout → replay-window pipeline.
    """

    def __init__(
        self,
        proprio_dim: int = 4,
        action_dim: int = 2,
        target: Tensor | None = None,
        max_steps: int = 30,
    ) -> None:
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        if action_dim > proprio_dim:
            raise ValueError("action_dim must be <= proprio_dim for MockChunkEnv")
        self.target = target if target is not None else torch.ones(action_dim)
        self.max_steps = max_steps
        self._proprio = torch.zeros(proprio_dim)
        self._t = 0

    def reset(self) -> Tensor:
        self._proprio = torch.zeros(self.proprio_dim)
        self._t = 0
        return self._proprio.clone()

    def step(self, action: Tensor) -> tuple[Tensor, float, bool]:
        self._proprio[: self.action_dim] = self._proprio[: self.action_dim] + 0.1 * action
        self._t += 1
        reward = 0.0
        done = False
        if torch.allclose(self._proprio[: self.action_dim], self.target, atol=1e-2):
            reward = 1.0
            done = True
        if self._t >= self.max_steps:
            done = True
        return self._proprio.clone(), reward, done


class ChunkedRolloutRunner:
    """Open-loop chunked rollout producing lerobot-compatible replay windows.

    At each chunk boundary the runner queries ``machine_a_fn(obs)`` for the
    frozen VLA reference + RL token, asks the actor for a chunk, and executes
    ``chunk_len`` actions open-loop via ``env_step_fn``. Per-tick
    :class:`StepRecord` entries are accumulated and sliced into
    :class:`ReplayWindow` objects at episode end (reusing lerobot's
    ``build_replay_windows``).
    """

    def __init__(
        self,
        chunk_len: int,
        action_dim: int,
        token_dim: int,
        proprio_dim: int,
        device: str = "cpu",
        stride: int = 0,
        discount: float = 0.99,
    ) -> None:
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.proprio_dim = proprio_dim
        self.device = device
        self.stride = stride
        self.discount = discount

    def run_episode(
        self,
        machine_a_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
        actor_chunk_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        env_reset_fn: Callable[[], Tensor],
        env_step_fn: Callable[[Tensor], tuple[Tensor, float, bool]],
        max_chunks: int = 10,
    ) -> list:
        """Run one episode and return ``list[ReplayWindow]``."""
        proprio = env_reset_fn()
        trace: list[StepRecord] = []
        t = 0
        for cid in range(max_chunks):
            z_rl, ref_chunk = machine_a_fn(proprio)
            # z_rl: [token_dim], ref_chunk: [C, d]
            chunk_flat = actor_chunk_fn(z_rl, proprio, ref_chunk.reshape(-1))
            chunk = chunk_flat.reshape(self.chunk_len, self.action_dim)
            for j in range(self.chunk_len):
                a = chunk[j]
                next_proprio, reward, done = env_step_fn(a)
                trace.append(
                    StepRecord(
                        t=t,
                        chunk_id=cid,
                        t_in_chunk=j,
                        z_rl=z_rl.detach().cpu(),
                        ref_chunk=ref_chunk.detach().cpu(),
                        proprio=proprio.detach().cpu(),
                        executed=a.detach().cpu(),
                        reward=float(reward),
                        done=bool(done),
                        source=SOURCE_RL,
                        human_action=None,
                    )
                )
                t += 1
                proprio = next_proprio
                if done:
                    return build_replay_windows(
                        trace, self.chunk_len, stride=self.stride, discount=self.discount
                    )
        return build_replay_windows(trace, self.chunk_len, stride=self.stride, discount=self.discount)

    def run_episode_with_env(
        self,
        machine_a_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
        actor_chunk_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        env: MockChunkEnv,
        max_chunks: int = 10,
    ):
        """Convenience wrapper using a :class:`MockChunkEnv`."""
        return self.run_episode(
            machine_a_fn,
            actor_chunk_fn,
            env_reset_fn=env.reset,
            env_step_fn=env.step,
            max_chunks=max_chunks,
        )


def make_mock_machine_a(token_dim: int, chunk_len: int, action_dim: int, seed: int = 0):
    """Return a deterministic ``machine_a_fn`` emitting fixed z_rl + ref chunks."""
    g = torch.Generator().manual_seed(seed)

    def machine_a_fn(proprio: Tensor) -> tuple[Tensor, Tensor]:
        z = torch.randn(token_dim, generator=g)
        ref = torch.randn(chunk_len, action_dim, generator=g)
        return z, ref

    return machine_a_fn


def make_mock_actor_chunk(actor_chunk_fn=None):
    """Default mock actor: returns a zero chunk (SOURCE_BASE semantics).

    For conformance tests the actor's *content* doesn't matter — only the
    pipeline shape. Callers can pass their own ``actor_chunk_fn``.
    """

    def default_fn(z_rl: Tensor, proprio: Tensor, ref_flat: Tensor) -> Tensor:
        return ref_flat.clone()

    return actor_chunk_fn or default_fn


def run_mock_episode(
    chunk_len: int = 3,
    action_dim: int = 2,
    token_dim: int = 8,
    proprio_dim: int = 4,
    max_chunks: int = 4,
    stride: int = 0,
) -> list:
    """One-call helper: run a fully mock episode and return replay windows."""
    env = MockChunkEnv(proprio_dim=proprio_dim, action_dim=action_dim, max_steps=chunk_len * max_chunks)
    runner = ChunkedRolloutRunner(
        chunk_len=chunk_len,
        action_dim=action_dim,
        token_dim=token_dim,
        proprio_dim=proprio_dim,
        stride=stride,
    )
    machine_a = make_mock_machine_a(token_dim, chunk_len, action_dim)
    actor = make_mock_actor_chunk()
    return runner.run_episode_with_env(machine_a, actor, env, max_chunks=max_chunks)
