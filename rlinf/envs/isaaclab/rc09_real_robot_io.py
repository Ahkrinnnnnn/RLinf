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

"""Unit conversion between Isaac Sim (rad) and CRP real-robot LeRobot SFT (deg, GOT0).

LeRobot ``insert_the_blue_tube`` / ``crp_arm`` logs:
- ``observation.state`` / arm joints: **degrees** (j1.pos … j6.pos)
- ``action`` gripper: **GOT0** register in [0, 1000] (0=closed, 1000=open)

Isaac Lab RC09 task uses radians for joint pos/vel limits and absolute joint targets.

Safety clipping uses RC09-05 **product manual** joint limits (degrees) and CRP GOT0
range — not dataset quantiles. Pi0.5 quantile normalization is unchanged (SFT stats).
"""

from __future__ import annotations

import math

import torch

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

# RC09-05 manual limits (deg), j1…j6 — keep in sync with rc09_robot_cfg.py
_ARM_JOINT_MIN_DEG = (-360.0, -360.0, -78.0, -210.0, -360.0, -210.0)
_ARM_JOINT_MAX_DEG = (360.0, 360.0, 256.0, 210.0, 360.0, 210.0)

GRIPPER_GOT0_MIN = 0.0
GRIPPER_GOT0_MAX = 1000.0
# Manual semantics: 0=closed, 1000=open (CRP GOT0); midpoint for Isaac binary gripper.
GRIPPER_GOT0_OPEN_THRESHOLD = 500.0


def _arm_limit_tensors(
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.tensor(_ARM_JOINT_MIN_DEG, device=ref.device, dtype=ref.dtype)
    maxs = torch.tensor(_ARM_JOINT_MAX_DEG, device=ref.device, dtype=ref.dtype)
    return mins, maxs


def clip_policy_joints_to_limits(joint_pos_deg: torch.Tensor) -> torch.Tensor:
    """Clip arm joint positions (deg) to RC09-05 manual limits."""
    mins, maxs = _arm_limit_tensors(joint_pos_deg)
    return joint_pos_deg.clamp(mins, maxs)


def clip_policy_action_to_limits(action: torch.Tensor) -> torch.Tensor:
    """Clip 7D policy action (6 joints deg + GOT0 gripper) to manual / CRP ranges."""
    if action.shape[-1] < 7:
        raise ValueError(f"Expected action dim >= 7, got {action.shape[-1]}")
    out = action.clone()
    out[..., :6] = clip_policy_joints_to_limits(out[..., :6])
    out[..., 6] = out[..., 6].clamp(GRIPPER_GOT0_MIN, GRIPPER_GOT0_MAX)
    return out


def sim_joint_pos_to_policy(joint_pos_rad: torch.Tensor) -> torch.Tensor:
    """Convert Isaac arm joint positions (rad) to LeRobot policy state (deg, clipped)."""
    return clip_policy_joints_to_limits(joint_pos_rad * RAD2DEG)


def policy_arm_joints_to_sim(
    joint_pos_deg: torch.Tensor,
    *,
    joint_signs: tuple[float, float, float, float, float, float] | None = None,
) -> torch.Tensor:
    """Convert policy arm joint targets (deg) to Isaac absolute joint targets (rad)."""
    clipped = clip_policy_joints_to_limits(joint_pos_deg)
    rad = clipped * DEG2RAD
    if joint_signs is not None:
        signs = torch.tensor(joint_signs, device=rad.device, dtype=rad.dtype)
        rad = rad * signs
    return rad


def policy_gripper_to_sim_binary(gripper_got0: torch.Tensor) -> torch.Tensor:
    """Map CRP GOT0 [0, 1000] to Isaac binary gripper command (+1 open / -1 close)."""
    got0 = gripper_got0.clamp(GRIPPER_GOT0_MIN, GRIPPER_GOT0_MAX)
    return torch.where(
        got0 >= GRIPPER_GOT0_OPEN_THRESHOLD,
        torch.ones_like(got0),
        -torch.ones_like(got0),
    )


def policy_action_to_sim(
    action: torch.Tensor,
    *,
    joint_signs: tuple[float, float, float, float, float, float] | None = None,
) -> torch.Tensor:
    """Convert 7D LeRobot policy action to Isaac RC09 action tensor."""
    clipped = clip_policy_action_to_limits(action)
    out = clipped.clone()
    out[..., :6] = policy_arm_joints_to_sim(clipped[..., :6], joint_signs=joint_signs)
    out[..., 6] = policy_gripper_to_sim_binary(clipped[..., 6])
    return out
