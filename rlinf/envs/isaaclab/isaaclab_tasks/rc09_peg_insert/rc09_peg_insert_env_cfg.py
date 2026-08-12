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

"""Environment configuration for RC09 tabletop visuomotor subtasks.

Scene: CAD parts from ``scene_assets/001`` (peg, tube, base, middle, cover).
Three skills share the scene; active one is ``RC09_SUBTASK`` (see ``subtasks.py``).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp
from .camera_calibration import load_rc09_calibrated_cameras, make_camera_cfg
from .camera_calibration import RC09_TCP_OFFSET_POS_M
from .rc09_robot_cfg import (
    RC09_ARM_JOINTS,
    RC09_EE_BODY_NAME,
    RC09_GRIPPER_CLOSE_POS,
    RC09_GRIPPER_FINGER_LINKS,
    RC09_GRIPPER_FORCE_LIMIT_N,
    RC09_GRIPPER_JOINTS,
    RC09_GRIPPER_OPEN_POS,
    RC09_ROBOT_CFG,
)
from .subtasks import resolve_active_subtask

_ACTIVE_SUBTASK = resolve_active_subtask()
TASK_PROMPT = _ACTIVE_SUBTASK.prompt

# CAD sizes from scene_assets/001 (mm -> m), geometric-center origin.
PEG_RADIUS = 0.014  # 圆柱 OD ≈ 28 mm
PEG_HEIGHT = 0.09  # 圆柱 height 90 mm
TUBE_OUTER_RADIUS = 0.0249  # 空心圆柱 OD ≈ 49.8 mm
TUBE_HEIGHT = 0.07  # 空心圆柱 height 70 mm
BASE_HEIGHT = 0.05  # 大方形底座
MIDDLE_HEIGHT = 0.07  # 中间件
COVER_HEIGHT = 0.04  # 顶盖
# After flipping base (pocket opens +Z), cavity floor is 20 mm above table.
BASE_POCKET_FLOOR_Z = 0.02
GROUND_Z = 0.0
# Lab layout: 74.2 cm tabletop, robot base flange at 1.1 m.
# World XY: table ~50 cm forward of base (sim world). Pendant user-frame is
# intentionally unused — joint datasets do not depend on it.
TABLE_TOP_Z = 0.742
TABLE_THICKNESS = 0.05
TABLE_SIZE = (0.424, 1.60, TABLE_THICKNESS)
ROBOT_BASE_Z = 1.1
TABLE_CENTER_XY = (0.50, 0.0, 0.0)
BASE_INIT_XY = (TABLE_CENTER_XY[0], TABLE_CENTER_XY[1])
# Side slots for parts that are not yet assembled.
_SIDE_PEG_XY = (TABLE_CENTER_XY[0], 0.22)
_SIDE_TUBE_XY = (TABLE_CENTER_XY[0], -0.22)
_SIDE_MIDDLE_XY = (TABLE_CENTER_XY[0], 0.28)
COVER_INIT_XY = (TABLE_CENTER_XY[0], 0.55)

# --- Start poses depend on which prior skills are already done ---
if _ACTIVE_SUBTASK.middle_pose == "nested_in_base":
    MIDDLE_INIT_XY = BASE_INIT_XY
    MIDDLE_INIT_Z = TABLE_TOP_Z + BASE_POCKET_FLOOR_Z + MIDDLE_HEIGHT / 2
else:
    # stack_base: purple still beside empty black tray.
    MIDDLE_INIT_XY = _SIDE_MIDDLE_XY
    MIDDLE_INIT_Z = TABLE_TOP_Z + MIDDLE_HEIGHT / 2

if _ACTIVE_SUBTASK.tube_pose == "seated_in_purple":
    # insert_rod: tube already plugged into purple (same XY as base stack).
    TUBE_INIT_XY = BASE_INIT_XY
    # Sit on purple top (~ pocket floor + middle height), not on the bare table.
    TUBE_INIT_Z = TABLE_TOP_Z + BASE_POCKET_FLOOR_Z + MIDDLE_HEIGHT + TUBE_HEIGHT / 2
else:
    TUBE_INIT_XY = _SIDE_TUBE_XY
    TUBE_INIT_Z = TABLE_TOP_Z + TUBE_HEIGHT / 2

if _ACTIVE_SUBTASK.peg_pose == "seated_in_tube":
    PEG_INIT_XY = TUBE_INIT_XY
    PEG_INIT_Z = TUBE_INIT_Z + TUBE_HEIGHT / 2
else:
    PEG_INIT_XY = _SIDE_PEG_XY
    PEG_INIT_Z = TABLE_TOP_Z + PEG_HEIGHT / 2

# 180° about X so the tray pocket opens upward (CAD exports opening downward).
BASE_INIT_ROT = (0.0, 1.0, 0.0, 0.0)

_SCENE_ASSET_DIR = (
    Path(__file__).resolve().parent / "assets" / "scene_001"
)
_PEG_USD = str((_SCENE_ASSET_DIR / "peg.usda").resolve())
_TUBE_USD = str((_SCENE_ASSET_DIR / "tube.usda").resolve())
_BASE_USD = str((_SCENE_ASSET_DIR / "base.usda").resolve())
_MIDDLE_USD = str((_SCENE_ASSET_DIR / "middle.usda").resolve())
_COVER_USD = str((_SCENE_ASSET_DIR / "cover.usda").resolve())

_CAMS = load_rc09_calibrated_cameras()
_TABLE_SPEC, _TABLE_BIND = _CAMS["table_cam"]
_WRIST_SPEC, _WRIST_BIND = _CAMS["wrist_cam"]
# Intrinsics + extrinsics come only from hand-eye calibration (no pose override).
_TABLE_CAM_CFG = make_camera_cfg(
    _TABLE_SPEC,
    _TABLE_BIND,
    prim_path="{ENV_REGEX_NS}/Robot/" + _TABLE_BIND.parent_link + "/table_cam",
)
_WRIST_CAM_CFG = make_camera_cfg(
    _WRIST_SPEC,
    _WRIST_BIND,
    prim_path="{ENV_REGEX_NS}/Robot/" + _WRIST_BIND.parent_link + "/wrist_cam",
)


def _table_object_cfg(
    *,
    prim_name: str,
    usd_path: str,
    xy: tuple[float, float],
    height: float,
    color: tuple[float, float, float],
    kinematic: bool,
    mass: float,
    pos_z: float | None = None,
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> RigidObjectCfg:
    """Spawn a CAD mesh (origin = geometric center). Default Z sits on the tabletop."""
    if pos_z is None:
        pos_z = TABLE_TOP_Z + height / 2
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], pos_z),
            rot=rot,
        ),
    )


@configclass
class RC09PegInsertSceneCfg(InteractiveSceneCfg):
    """Tabletop scene: RC09 arm + scene_assets/001 CAD parts + cameras."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)),
    )

    # Plain wooden tabletop (axis-aligned), long side across Y — in front of the arm.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.36, 0.20),
                roughness=0.7,
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z - TABLE_THICKNESS / 2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    robot = RC09_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=RC09_ROBOT_CFG.init_state.replace(pos=(0.0, 0.0, ROBOT_BASE_Z)),
    )

    # scene_assets/001 — start layout = prior assembly state (see ``subtasks.py``).
    peg = _table_object_cfg(
        prim_name="Peg",
        usd_path=_PEG_USD,
        xy=PEG_INIT_XY,
        height=PEG_HEIGHT,
        color=(0.85, 0.15, 0.55),  # pink rod
        kinematic=not _ACTIVE_SUBTASK.peg_dynamic,
        mass=0.08,
        pos_z=PEG_INIT_Z,
    )
    tube = _table_object_cfg(
        prim_name="Tube",
        usd_path=_TUBE_USD,
        xy=TUBE_INIT_XY,
        height=TUBE_HEIGHT,
        color=(0.05, 0.40, 0.95),  # blue tube
        kinematic=not _ACTIVE_SUBTASK.tube_dynamic,
        mass=0.25,
        pos_z=TUBE_INIT_Z,
    )
    base = _table_object_cfg(
        prim_name="Base",
        usd_path=_BASE_USD,
        xy=BASE_INIT_XY,
        height=BASE_HEIGHT,
        color=(0.05, 0.05, 0.05),  # black tray
        kinematic=True,
        mass=0.5,
        rot=BASE_INIT_ROT,
    )
    middle = _table_object_cfg(
        prim_name="Middle",
        usd_path=_MIDDLE_USD,
        xy=MIDDLE_INIT_XY,
        height=MIDDLE_HEIGHT,
        color=(0.55, 0.15, 0.75),  # purple insert / "purple base"
        kinematic=not _ACTIVE_SUBTASK.middle_dynamic,
        mass=0.2,
        pos_z=MIDDLE_INIT_Z,
    )
    cover = _table_object_cfg(
        prim_name="Cover",
        usd_path=_COVER_USD,
        xy=COVER_INIT_XY,
        height=COVER_HEIGHT,
        color=(0.10, 0.62, 0.55),  # teal lid (side clutter)
        kinematic=True,
        mass=0.15,
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + RC09_EE_BODY_NAME,
                name="ee",
                # Approx TCP along EE +Z (20 cm). Not the pendant tool offset.
                offset=OffsetCfg(pos=RC09_TCP_OFFSET_POS_M),
            ),
        ],
    )

    # FrameTransformer only tracks rigid bodies — not Camera prims. Visualize the
    # same offsets used by table_cam / wrist_cam on their parent links instead.
    # RGB axes: X=red, Y=green, Z=blue (ROS optical Z = look direction).
    camera_frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/RC09CameraFrames",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.12, 0.12, 0.12),
                ),
                "connecting_line": sim_utils.CylinderCfg(
                    radius=0.001,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.9, 0.1), roughness=1.0
                    ),
                ),
            },
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + _TABLE_BIND.parent_link,
                name="table_cam",
                offset=OffsetCfg(pos=_TABLE_CAM_CFG.offset.pos, rot=_TABLE_CAM_CFG.offset.rot),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + _WRIST_BIND.parent_link,
                name="wrist_cam",
                offset=OffsetCfg(pos=_WRIST_CAM_CFG.offset.pos, rot=_WRIST_CAM_CFG.offset.rot),
            ),
        ],
    )

    contact_force_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + RC09_GRIPPER_FINGER_LINKS[0],
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )
    contact_force_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + RC09_GRIPPER_FINGER_LINKS[1],
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    table_cam = _TABLE_CAM_CFG
    wrist_cam = _WRIST_CAM_CFG

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """7-dim policy action: 6 absolute arm joint positions (rad) + 1 binary gripper."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=RC09_ARM_JOINTS,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=RC09_GRIPPER_JOINTS,
        open_command_expr={name: RC09_GRIPPER_OPEN_POS for name in RC09_GRIPPER_JOINTS},
        close_command_expr={name: RC09_GRIPPER_CLOSE_POS for name in RC09_GRIPPER_JOINTS},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=mdp.arm_joint_pos)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        peg_pos = ObsTerm(func=mdp.peg_pos)
        tube_pos = ObsTerm(func=mdp.tube_pos)
        table_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    insert_success = RewTerm(
        func=mdp.peg_insert_success,
        weight=1.0,
        params={"xy_threshold": 0.008, "z_margin": 0.01},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insert_success = DoneTerm(
        func=mdp.peg_insert_success_consecutive,
        params={"num_steps": 5, "xy_threshold": 0.008, "z_margin": 0.01},
    )
    excessive_gripper_force = DoneTerm(
        func=mdp.gripper_excessive_force,
        params={"force_threshold": RC09_GRIPPER_FORCE_LIMIT_N},
    )


@configclass
class EventCfgBase:
    """Task reset events (always active)."""

    reset_success_streak = EventTerm(
        func=mdp.reset_success_streak,
        mode="reset",
    )

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Fixed home (dataset-aligned). Joint DR is added in EventCfg.
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_peg = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Workpiece on table may jitter; seated fixtures stay fixed.
            "pose_range": (
                {
                    "x": (-0.05, 0.05),
                    "y": (-0.06, 0.06),
                    "z": (0.0, 0.0),
                    "yaw": (-0.3, 0.3),
                }
                if _ACTIVE_SUBTASK.peg_pose == "on_table" and _ACTIVE_SUBTASK.peg_dynamic
                else {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
            ),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("peg"),
        },
    )

    reset_tube = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": (
                {
                    "x": (-0.04, 0.04),
                    "y": (-0.04, 0.04),
                    "z": (0.0, 0.0),
                    "yaw": (-0.15, 0.15),
                }
                if _ACTIVE_SUBTASK.tube_pose == "on_table" and _ACTIVE_SUBTASK.tube_dynamic
                else {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
            ),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tube"),
        },
    )

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("base"),
        },
    )

    reset_middle = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Beside-base workpiece may jitter; nested purple stays locked to base.
            "pose_range": (
                {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (0.0, 0.0),
                    "yaw": (-0.2, 0.2),
                }
                if _ACTIVE_SUBTASK.middle_pose == "beside_base" and _ACTIVE_SUBTASK.middle_dynamic
                else {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
            ),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("middle"),
        },
    )

    reset_cover = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cover"),
        },
    )


@configclass
class EventCfg(EventCfgBase):
    """Reset events plus VLA-style domain randomization."""

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # --- Visual domain randomization (requires replicate_physics=False) ---
    randomize_light = EventTerm(
        func=mdp.randomize_scene_lighting,
        mode="reset",
        params={
            "intensity_range": (1500.0, 4000.0),
            "color_variation": 0.35,
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
            ],
            "default_intensity": 2500.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": "",
        },
    )

    randomize_table_visual_material = EventTerm(
        func=base_mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "event_name": "rc09_table_texture_randomizer",
            "texture_paths": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
        },
    )

    randomize_peg_color = EventTerm(
        func=base_mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("peg"),
            "mesh_name": "geometry",
            "event_name": "rc09_peg_color_randomizer",
            # Keep hues in the pink / magenta family so the language prompt stays valid.
            "colors": {"r": (0.75, 1.0), "g": (0.25, 0.65), "b": (0.45, 0.95)},
        },
    )

    randomize_tube_color = EventTerm(
        func=base_mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tube"),
            "mesh_name": "geometry",
            "event_name": "rc09_tube_color_randomizer",
            "colors": {"r": (0.0, 0.25), "g": (0.20, 0.55), "b": (0.60, 1.0)},
        },
    )

    # Extrinsic-only DR
    randomize_table_cam_extrinsics = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("table_cam"),
            "position_noise": ((-0.003, 0.003), (-0.003, 0.003), (-0.003, 0.003)),
            "euler_noise_deg": ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
        },
    )

    # --- Physics domain randomization ---
    # Use mode="reset" (not "startup"): EventManager is constructed before sim.play(),
    # so ManagerTermBase classes are only instantiated on the PLAY callback. Calling
    # apply(mode="startup") before that treats the class as a plain function and crashes.
    peg_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("peg"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    peg_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("peg"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class RC09PegInsertVisuomotorEnvCfg(ManagerBasedRLEnvCfg):
    """RC09 peg-insert visuomotor RL env with optional domain randomization."""

    enable_domain_randomization: bool = True

    scene: RC09PegInsertSceneCfg = RC09PegInsertSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    image_obs_list = ["table_cam", "wrist_cam"]

    def set_domain_randomization(self, enabled: bool) -> None:
        """Enable/disable DR after construction (also swaps the events table)."""
        self.enable_domain_randomization = enabled
        self.events = EventCfg() if enabled else EventCfgBase()

    def __post_init__(self):
        self.set_domain_randomization(self.enable_domain_randomization)

        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0
        # Viewport / video camera (not table_cam / wrist_cam).
        self.viewer.eye = (2.0, 1.6, TABLE_TOP_Z + 1.0)
        self.viewer.lookat = (TABLE_CENTER_XY[0], 0.0, TABLE_TOP_Z)
        self.viewer.resolution = (1280, 720)
        self.num_rerenders_on_reset = 2
        self.scene.contact_force_left.update_period = self.sim.dt
        self.scene.contact_force_right.update_period = self.sim.dt
