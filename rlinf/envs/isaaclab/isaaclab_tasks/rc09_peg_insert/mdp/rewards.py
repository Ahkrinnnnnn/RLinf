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

"""Reward terms for RC09 peg-insert task."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import peg_inserted_z_ok, peg_to_tube_xy_dist


def peg_insert_success(
    env,
    xy_threshold: float = 0.008,
    z_margin: float = 0.01,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
) -> torch.Tensor:
    """Sparse success when purple peg is inserted into blue tube."""
    xy_dist = peg_to_tube_xy_dist(env, peg_cfg=peg_cfg, tube_cfg=tube_cfg)
    z_ok = peg_inserted_z_ok(env, peg_cfg=peg_cfg, tube_cfg=tube_cfg, z_margin=z_margin)
    return ((xy_dist < xy_threshold) & z_ok).float()


def peg_insert_shaping(
    env,
    std: float = 0.06,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
) -> torch.Tensor:
    """Dense shaping: encourage peg to align with tube opening in XY."""
    xy_dist = peg_to_tube_xy_dist(env, peg_cfg=peg_cfg, tube_cfg=tube_cfg)
    return torch.exp(-xy_dist / std)


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
