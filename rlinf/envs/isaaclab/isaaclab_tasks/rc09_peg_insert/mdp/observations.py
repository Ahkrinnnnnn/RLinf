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

"""Observation helpers for RC09 peg-insert task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

from ..rc09_robot_cfg import RC09_EE_BODY_NAME, RC09_GRIPPER_JOINTS

TUBE_HEIGHT = 0.06


def ee_frame_pos(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: Articulation = env.scene[ee_frame_cfg.name]
    return ee_frame.data.root_pos_w - env.scene.env_origins


def ee_frame_quat(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: Articulation = env.scene[ee_frame_cfg.name]
    return ee_frame.data.root_quat_w


def gripper_pos(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids = [robot.joint_names.index(name) for name in RC09_GRIPPER_JOINTS]
    positions = robot.data.joint_pos[:, joint_ids]
    normalized = positions / 0.03
    return normalized.mean(dim=-1, keepdim=True)


def peg_pos(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    peg = env.scene[peg_cfg.name]
    return peg.data.root_pos_w - env.scene.env_origins


def tube_pos(
    env,
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
) -> torch.Tensor:
    tube = env.scene[tube_cfg.name]
    return tube.data.root_pos_w - env.scene.env_origins


def image(
    env,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    camera: Camera = env.scene[sensor_cfg.name]
    images = camera.data.output[data_type]
    if normalize:
        return images.float() / 255.0
    return images


def peg_to_tube_xy_dist(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
) -> torch.Tensor:
    """XY distance between peg center and tube center."""
    peg = env.scene[peg_cfg.name]
    tube = env.scene[tube_cfg.name]
    peg_xy = peg.data.root_pos_w[:, :2]
    tube_xy = tube.data.root_pos_w[:, :2]
    return torch.linalg.norm(peg_xy - tube_xy, dim=-1)


def peg_inserted_z_ok(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
    z_margin: float = 0.01,
) -> torch.Tensor:
    """True when peg center z is within the tube vertical range."""
    peg = env.scene[peg_cfg.name]
    tube = env.scene[tube_cfg.name]
    peg_z = peg.data.root_pos_w[:, 2]
    tube_z = tube.data.root_pos_w[:, 2]
    tube_bottom = tube_z - TUBE_HEIGHT / 2
    tube_top = tube_z + TUBE_HEIGHT / 2
    return (peg_z >= tube_bottom - z_margin) & (peg_z <= tube_top + z_margin)
