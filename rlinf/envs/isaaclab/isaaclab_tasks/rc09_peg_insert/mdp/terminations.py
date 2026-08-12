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

"""Termination terms for RC09 peg-insert task."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ..rc09_robot_cfg import RC09_GRIPPER_FORCE_LIMIT_N
from .rewards import peg_insert_success


def time_out(env):
    """Episode timeout."""
    return env.episode_length_buf >= env.max_episode_length - 1


def _get_success_streak(env) -> torch.Tensor:
    if not hasattr(env, "_rc09_success_streak"):
        env._rc09_success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._rc09_success_streak


def reset_success_streak(env, env_ids) -> None:
    """Clear consecutive-success counter on episode reset."""
    streak = _get_success_streak(env)
    streak[env_ids] = 0


def peg_insert_success_consecutive(
    env,
    num_steps: int = 5,
    xy_threshold: float = 0.008,
    z_margin: float = 0.01,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
) -> torch.Tensor:
    """Terminate only after ``num_steps`` consecutive successful insert checks."""
    success = peg_insert_success(
        env,
        xy_threshold=xy_threshold,
        z_margin=z_margin,
        peg_cfg=peg_cfg,
        tube_cfg=tube_cfg,
    ).bool()
    streak = _get_success_streak(env)
    streak[:] = torch.where(success, streak + 1, torch.zeros_like(streak))
    return streak >= num_steps


def _max_contact_force_norm(contact_sensor: ContactSensor) -> torch.Tensor:
    """Return per-env max contact force norm (N) across sensor bodies and history."""
    forces = contact_sensor.data.net_forces_w_history
    if forces is None:
        forces = contact_sensor.data.net_forces_w.unsqueeze(1)
    # (num_envs, history, num_bodies, 3) -> (num_envs,)
    per_body = torch.linalg.vector_norm(forces, dim=-1)
    return per_body.amax(dim=(1, 2))


def gripper_excessive_force(
    env,
    force_threshold: float = RC09_GRIPPER_FORCE_LIMIT_N,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_force_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_force_right"),
) -> torch.Tensor:
    """Terminate when either gripper finger contact force exceeds ``force_threshold``."""
    left_sensor: ContactSensor = env.scene[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene[right_sensor_cfg.name]
    left_force = _max_contact_force_norm(left_sensor)
    right_force = _max_contact_force_norm(right_sensor)
    return (left_force > force_threshold) | (right_force > force_threshold)
