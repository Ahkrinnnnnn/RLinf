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

"""RC09 robot articulation configuration for Isaac Lab."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Resolve workspace root: .../RLinf/rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/rc09_robot_cfg.py
def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_urdf_path() -> Path:
    return _workspace_root() / "RC09-05_urdf.SLDASM/urdf/RC09-05_urdf.SLDASM.urdf"


_DEFAULT_URDF = _default_urdf_path()

RC09_ARM_JOINTS = [
    "J1_Joint",
    "J2_Joint",
    "J3_joint",
    "J4_Joint",
    "J5_Joint",
    "J6_Joint",
]
RC09_GRIPPER_JOINTS = ["right_gripper_joint", "left_gripper_joint"]
RC09_EE_BODY_NAME = "PGC-300-60"

# SolidWorks export leaves arm joint limits at 0; use reasonable defaults in sim.
RC09_DEFAULT_ARM_POS = {
    "J1_Joint": 0.0,
    "J2_Joint": -0.8,
    "J3_joint": 1.2,
    "J4_Joint": 0.0,
    "J5_Joint": 0.8,
    "J6_Joint": 0.0,
    "right_gripper_joint": 0.015,
    "left_gripper_joint": 0.015,
}


def resolve_rc09_urdf_path() -> str:
    """Resolve RC09 URDF path from env var or workspace default."""
    env_path = os.environ.get("RC09_URDF_PATH")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str(_DEFAULT_URDF.resolve())


def get_rc09_urdf_asset_path() -> str:
    """Return URDF path, preferring mesh-patched copy when available."""
    urdf_path = Path(resolve_rc09_urdf_path())
    patched = urdf_path.with_name("RC09-05_urdf.SLDASM.isaac.urdf")
    if patched.is_file():
        return str(patched.resolve())
    return str(urdf_path.resolve())


RC09_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=get_rc09_urdf_asset_path(),
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=True,
        link_density=1000.0,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=400.0,
                damping=40.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=RC09_DEFAULT_ARM_POS,
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=RC09_ARM_JOINTS,
            stiffness=400.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=RC09_GRIPPER_JOINTS,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
