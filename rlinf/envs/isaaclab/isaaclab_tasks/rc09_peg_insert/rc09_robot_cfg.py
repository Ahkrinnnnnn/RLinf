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

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .subtasks import home_joint_pos_rad, resolve_active_subtask


# Resolve workspace root: .../RLinf/rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/rc09_robot_cfg.py
def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_urdf_path() -> Path:
    return _workspace_root() / "RC09-05_urdf.SLDASM/urdf/RC09-05_urdf.SLDASM.urdf"


_DEFAULT_URDF = _default_urdf_path()

# Product manual (RC09-05) kinematic specs.
RC09_MAX_PAYLOAD_KG = 5.0
RC09_MAX_REACH_M = 0.906

_DEG = math.pi / 180.0

RC09_ARM_JOINTS = [
    "J1_Joint",
    "J2_Joint",
    "J3_joint",
    "J4_Joint",
    "J5_Joint",
    "J6_Joint",
]
RC09_GRIPPER_JOINTS = ["right_gripper_joint", "left_gripper_joint"]
RC09_GRIPPER_FINGER_LINKS = ["left_gripper", "right_gripper"]
# URDF link is "PGC-300-60"; Isaac URDF importer rewrites "-" to "_" for USD prim paths.
RC09_EE_BODY_NAME = "PGC_300_60"

# Prismatic finger joints (m): 0 = fully open, 0.03 = fully closed.
RC09_GRIPPER_OPEN_POS = 0.0
RC09_GRIPPER_CLOSE_POS = 0.03

# Terminate when either finger contact force exceeds this limit (N).
RC09_GRIPPER_FORCE_LIMIT_N = 40.0

# Joint position limits (degrees) from product manual.
RC09_ARM_JOINT_POS_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "J1_Joint": (-360.0, 360.0),
    "J2_Joint": (-360.0, 360.0),
    "J3_joint": (-78.0, 256.0),
    "J4_Joint": (-210.0, 210.0),  # manual lists +210°; treat as symmetric ±210°
    "J5_Joint": (-360.0, 360.0),
    "J6_Joint": (-210.0, 210.0),
}

# Max joint speeds (deg/s) from product manual.
RC09_ARM_JOINT_VEL_LIMITS_DEG_S: dict[str, float] = {
    "J1_Joint": 225.0,
    "J2_Joint": 225.0,
    "J3_joint": 225.0,
    "J4_Joint": 250.0,
    "J5_Joint": 250.0,
    "J6_Joint": 250.0,
}

RC09_ARM_JOINT_POS_LIMITS = {
    name: (lower * _DEG, upper * _DEG) for name, (lower, upper) in RC09_ARM_JOINT_POS_LIMITS_DEG.items()
}
RC09_ARM_JOINT_VEL_LIMITS = {name: vel * _DEG for name, vel in RC09_ARM_JOINT_VEL_LIMITS_DEG_S.items()}

# --- Drive / actuator (joint-side SI units for Isaac ImplicitActuator) ---
# Mechanical reduction (all axes). Joint torque ≈ N * motor torque.
RC09_GEAR_RATIO = 101.0
# Motor rated torque (N·m): J1–J3 proximal group, J4–J6 distal (J6 assumed = J5).
RC09_MOTOR_RATED_TORQUE_NM = {
    "J1_Joint": 2.51,
    "J2_Joint": 2.51,
    "J3_joint": 2.51,
    "J4_Joint": 0.30,
    "J5_Joint": 0.30,
    "J6_Joint": 0.30,
}
RC09_ARM_EFFORT_LIMIT_SIM = {
    name: RC09_GEAR_RATIO * tau for name, tau in RC09_MOTOR_RATED_TORQUE_NM.items()
}

# PhysX / Isaac PD: τ ≈ kp*(q_des-q) + kd*(qd_des-qd), units N·m/rad and N·m·s/rad.
# CRP manual "position-loop P (rad/s)" and "velocity-loop P/I (rad/s)" are cascaded
# controller gains — NOT drop-in replacements for kp/kd. Use Franka-scale PD that
# can hold the arm against gravity (kp≈80 was too weak → arm collapses on reset).
RC09_ARM_JOINT_STIFFNESS: dict[str, float] = {
    "J1_Joint": 400.0,
    "J2_Joint": 400.0,
    "J3_joint": 400.0,
    "J4_Joint": 400.0,
    "J5_Joint": 400.0,
    "J6_Joint": 400.0,
}
RC09_ARM_JOINT_DAMPING: dict[str, float] = {
    "J1_Joint": 40.0,
    "J2_Joint": 40.0,
    "J3_joint": 40.0,
    "J4_Joint": 40.0,
    "J5_Joint": 40.0,
    "J6_Joint": 40.0,
}

# Active subtask home (deg→rad). Override with RC09_SUBTASK=insert_rod|insert_tube|stack_base.
# If the arm folds / wrist stares at the base, CRP↔URDF joint signs likely differ — set
# RC09_JOINT_SIGNS=1,-1,-1,1,1,1 (comma-separated) or pass --joint-signs to replay.
_ACTIVE_SUBTASK = resolve_active_subtask()


def _parse_joint_signs() -> tuple[float, float, float, float, float, float]:
    raw = os.environ.get("RC09_JOINT_SIGNS", "1,1,1,1,1,1")
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 6:
        raise ValueError(f"RC09_JOINT_SIGNS must have 6 floats, got {raw!r}")
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])


RC09_DEFAULT_ARM_POS = {
    **home_joint_pos_rad(_ACTIVE_SUBTASK, joint_signs=_parse_joint_signs()),
    "right_gripper_joint": RC09_GRIPPER_OPEN_POS,
    "left_gripper_joint": RC09_GRIPPER_OPEN_POS,
}


# SolidWorks URDF inertial export is ~2× the product-manual robot mass (~23 kg).
# Scale mass and inertia together so dynamics stay consistent (I ∝ m for uniform density).
RC09_URDF_INERTIAL_SCALE = 0.5


def resolve_rc09_urdf_path() -> str:
    """Resolve RC09 URDF path from env var or workspace default."""
    env_path = os.environ.get("RC09_URDF_PATH")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str(_DEFAULT_URDF.resolve())


def _patch_urdf_arm_limits(urdf_path: Path) -> None:
    """Sync arm + gripper joint lower/upper/velocity with product-manual constants.

    Only touches those attributes. Never writes ``effort`` (SolidWorks leave
    it at 0; Isaac torque comes from ``ImplicitActuatorCfg`` / ``joint_drive``).
    """
    if not urdf_path.is_file():
        return

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    updated = False

    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is None:
            continue

        if name in RC09_ARM_JOINT_POS_LIMITS_DEG:
            lower_deg, upper_deg = RC09_ARM_JOINT_POS_LIMITS_DEG[name]
            desired = {
                "lower": f"{lower_deg * _DEG:.8g}",
                "upper": f"{upper_deg * _DEG:.8g}",
                "velocity": f"{RC09_ARM_JOINT_VEL_LIMITS_DEG_S[name] * _DEG:.8g}",
            }
        elif name in RC09_GRIPPER_JOINTS:
            # Prismatic (m): 0 open → RC09_GRIPPER_CLOSE_POS closed.
            desired = {
                "lower": f"{RC09_GRIPPER_OPEN_POS:.8g}",
                "upper": f"{RC09_GRIPPER_CLOSE_POS:.8g}",
            }
        else:
            continue

        if all(limit.get(key) == value for key, value in desired.items()):
            continue
        for key, value in desired.items():
            limit.set(key, value)
        updated = True

    if updated:
        tree.write(urdf_path, encoding="unicode", xml_declaration=True)


def _patch_urdf_inertials(isaac_urdf: Path, source_urdf: Path, scale: float) -> None:
    """Rewrite Isaac URDF inertials from the SolidWorks source × ``scale``.

    Re-reads the unscaled source every call so the patch is idempotent.
    """
    if not isaac_urdf.is_file() or not source_urdf.is_file():
        return
    if abs(scale - 1.0) < 1e-12:
        return

    src_root = ET.parse(source_urdf).getroot()
    src_inertial: dict[str, ET.Element] = {}
    for link in src_root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if name and inertial is not None:
            src_inertial[name] = inertial

    tree = ET.parse(isaac_urdf)
    root = tree.getroot()
    updated = False
    for link in root.findall("link"):
        name = link.get("name")
        if name not in src_inertial:
            continue
        dst = link.find("inertial")
        src = src_inertial[name]
        if dst is None:
            continue

        src_mass = src.find("mass")
        dst_mass = dst.find("mass")
        if src_mass is not None and dst_mass is not None:
            new_mass = float(src_mass.get("value", "0")) * scale
            if dst_mass.get("value") != f"{new_mass:.8g}":
                dst_mass.set("value", f"{new_mass:.8g}")
                updated = True

        src_I = src.find("inertia")
        dst_I = dst.find("inertia")
        if src_I is not None and dst_I is not None:
            for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                raw = src_I.get(key)
                if raw is None:
                    continue
                new_val = float(raw) * scale
                if dst_I.get(key) != f"{new_val:.8g}":
                    dst_I.set(key, f"{new_val:.8g}")
                    updated = True

    if updated:
        # Marker so logs/debug can confirm the scale that was applied.
        root.set("rlinf_inertial_scale", f"{scale:g}")
        tree.write(isaac_urdf, encoding="unicode", xml_declaration=True)


def get_rc09_urdf_asset_path() -> str:
    """Prefer ``*.isaac.urdf`` (mesh paths for Isaac); sync limits + inertial scale."""
    urdf_path = Path(resolve_rc09_urdf_path())
    patched = urdf_path.with_name("RC09-05_urdf.SLDASM.isaac.urdf")
    if patched.is_file():
        _patch_urdf_arm_limits(patched)
        _patch_urdf_inertials(patched, urdf_path, RC09_URDF_INERTIAL_SCALE)
        return str(patched.resolve())
    return str(urdf_path.resolve())


RC09_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=get_rc09_urdf_asset_path(),
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=True,
        # Use URDF inertial masses (do not override with density*volume).
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=RC09_ARM_JOINT_STIFFNESS,
                damping=RC09_ARM_JOINT_DAMPING,
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
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=RC09_DEFAULT_ARM_POS,
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=RC09_ARM_JOINTS,
            stiffness=RC09_ARM_JOINT_STIFFNESS,
            damping=RC09_ARM_JOINT_DAMPING,
            effort_limit_sim=RC09_ARM_EFFORT_LIMIT_SIM,
            velocity_limit_sim=RC09_ARM_JOINT_VEL_LIMITS,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=RC09_GRIPPER_JOINTS,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
