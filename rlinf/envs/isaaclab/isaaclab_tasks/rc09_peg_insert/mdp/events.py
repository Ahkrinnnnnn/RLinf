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

"""Domain randomization events for RC09 peg-insert visuomotor training."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def domain_randomization_enabled(env: ManagerBasedEnv) -> bool:
    """Return whether visual/physics DR is active for this episode."""
    return getattr(env.cfg, "enable_domain_randomization", True)


def sample_random_color(base=(0.75, 0.75, 0.75), variation=0.1):
    """Sample an RGB color near *base* while preserving overall brightness."""
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    avg_offset = sum(offsets) / 3
    balanced_offsets = [offset - avg_offset for offset in offsets]
    return tuple(
        max(0.0, min(1.0, base_component + offset))
        for base_component, offset in zip(base, balanced_offsets)
    )


def randomize_scene_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    textures: list[str],
    default_intensity: float = 2500.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_texture: str = "",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize dome-light intensity, color, and optional HDR background."""
    if env_ids is None:
        return

    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    color_attr = light_prim.GetAttribute("inputs:color")
    texture_file_attr = light_prim.GetAttribute("inputs:texture:file")

    if not domain_randomization_enabled(env):
        intensity_attr.Set(default_intensity)
        color_attr.Set(default_color)
        texture_file_attr.Set(default_texture)
        return

    intensity_attr.Set(random.uniform(intensity_range[0], intensity_range[1]))
    color_attr.Set(sample_random_color(base=default_color, variation=color_variation))
    if textures:
        texture_file_attr.Set(random.choice(textures))


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _get_camera_defaults(env: ManagerBasedEnv, sensor_name: str, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    cache_attr = "_rc09_camera_defaults"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = getattr(env, cache_attr)
    if sensor_name not in cache:
        cache[sensor_name] = (
            camera.data.pos_w.clone(),
            camera.data.quat_w_ros.clone(),
        )
    return cache[sensor_name]


def randomize_camera_extrinsics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    position_noise: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    euler_noise_deg: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
):
    """Apply small pose perturbations to a fixed-mount camera on reset."""
    env_ids = _resolve_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    camera: Camera = env.scene[sensor_cfg.name]
    default_pos, default_quat = _get_camera_defaults(env, sensor_cfg.name, camera)

    pos = default_pos[env_ids].clone()
    quat = default_quat[env_ids].clone()

    if domain_randomization_enabled(env):
        for dim, (low, high) in enumerate(position_noise):
            pos[:, dim] += torch.empty(len(env_ids), device=env.device).uniform_(low, high)

        roll = torch.deg2rad(
            torch.empty(len(env_ids), device=env.device).uniform_(euler_noise_deg[0][0], euler_noise_deg[0][1])
        )
        pitch = torch.deg2rad(
            torch.empty(len(env_ids), device=env.device).uniform_(euler_noise_deg[1][0], euler_noise_deg[1][1])
        )
        yaw = torch.deg2rad(
            torch.empty(len(env_ids), device=env.device).uniform_(euler_noise_deg[2][0], euler_noise_deg[2][1])
        )
        delta_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        quat = math_utils.quat_mul(delta_quat, quat)

    camera.set_world_poses(pos, quat, env_ids.tolist(), convention="ros")
