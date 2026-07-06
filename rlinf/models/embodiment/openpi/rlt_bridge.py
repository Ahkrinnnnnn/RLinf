# Copyright 2026 The RLinf Authors.
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

"""Optional bridge between LeRobot RL Token (Stage 1) and RLinf OpenPI actor.

Path 2 from the RC09 integration plan: load a frozen ``RLTokenModel`` checkpoint
trained in LeRobot and expose compact RL-token features for the PPO value head.

Usage in Hydra config (``actor.model.openpi`` section)::

    use_rl_token: true
    rl_token_checkpoint: /path/to/rlt_pi05/checkpoints/last/pretrained_model

The bridge keeps action generation on the base OpenPI policy; only critic inputs
are augmented when ``use_rl_token`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class RLTokenBridgeConfig:
    """Configuration for optional RL Token loading."""

    enabled: bool = False
    checkpoint_path: str | None = None
    device: str = "cuda"


class RLTokenBridge(nn.Module):
    """Lazy-load LeRobot RL Token weights for critic feature augmentation."""

    def __init__(self, config: RLTokenBridgeConfig):
        super().__init__()
        self.config = config
        self._rlt_module: nn.Module | None = None
        self._backbone: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.checkpoint_path is not None

    def _lazy_init(self, base_policy: nn.Module) -> None:
        if self._rlt_module is not None:
            return
        if not self.enabled:
            return

        try:
            from lerobot.policies.rl_token.backbone import create_rlt_backbone
            from lerobot.policies.rl_token.rl_token import RLTokenConfig, RLTokenModel
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "RL Token bridge requires LeRobot with rl_token extras installed."
            ) from exc

        ckpt_dir = Path(self.config.checkpoint_path)
        config_path = ckpt_dir / "config.json"
        weights_path = ckpt_dir / "model.safetensors"
        if not weights_path.is_file():
            weights_path = ckpt_dir / "rlt_module.safetensors"

        rl_token_cfg = RLTokenConfig()
        if config_path.is_file():
            import json

            with config_path.open(encoding="utf-8") as f:
                raw = json.load(f)
            rlt_raw = raw.get("rlt_module", raw)
            for key, value in rlt_raw.items():
                if hasattr(rl_token_cfg, key):
                    setattr(rl_token_cfg, key, value)

        self._rlt_module = RLTokenModel(rl_token_cfg)
        if weights_path.is_file():
            state = load_file(str(weights_path))
            self._rlt_module.load_state_dict(state, strict=False)

        self._backbone = create_rlt_backbone(base_policy)
        self._rlt_module.eval()
        for param in self._rlt_module.parameters():
            param.requires_grad = False
        self._rlt_module.to(self.config.device)

    @torch.no_grad()
    def encode_from_openpi_batch(
        self,
        base_policy: nn.Module,
        openpi_batch: dict[str, Tensor],
    ) -> Tensor | None:
        """Return RL token embeddings for a batch, or ``None`` when disabled."""
        if not self.enabled:
            return None

        self._lazy_init(base_policy)
        assert self._backbone is not None and self._rlt_module is not None

        prefix = self._backbone.extract_prefix_embeddings(openpi_batch)
        embs = prefix.embeddings.to(dtype=torch.float32, device=self.config.device)
        mask = prefix.mask.to(device=self.config.device)
        return self._rlt_module.encode(embs, mask)

    def augment_critic_input(
        self,
        critic_features: Tensor,
        rl_token: Tensor | None,
    ) -> Tensor:
        """Concatenate RL token to critic features when available."""
        if rl_token is None:
            return critic_features
        if rl_token.ndim == 3:
            rl_token = rl_token.mean(dim=1)
        return torch.cat([critic_features, rl_token.to(critic_features.dtype)], dim=-1)


def build_rl_token_bridge(openpi_cfg: dict[str, Any]) -> RLTokenBridge:
    """Factory helper used by OpenPI actor initialization."""
    bridge_cfg = RLTokenBridgeConfig(
        enabled=bool(openpi_cfg.get("use_rl_token", False)),
        checkpoint_path=openpi_cfg.get("rl_token_checkpoint"),
        device=str(openpi_cfg.get("device", "cuda")),
    )
    return RLTokenBridge(bridge_cfg)
