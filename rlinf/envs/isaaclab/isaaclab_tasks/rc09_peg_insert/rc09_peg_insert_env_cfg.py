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

"""Environment configuration for RC09 peg-insert visuomotor task.

Scene: tabletop with a pink rod (peg) and a blue tube (socket).
Task: insert the pink rod into the blue tube.
Prompt: Insert the pink rod into the blue tube
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .rc09_robot_cfg import RC09_ARM_JOINTS, RC09_EE_BODY_NAME, RC09_GRIPPER_JOINTS, RC09_ROBOT_CFG

PEG_RADIUS = 0.012
PEG_HEIGHT = 0.05
TUBE_RADIUS = 0.018
TUBE_HEIGHT = 0.06
TABLE_TOP_Z = 0.0

TASK_PROMPT = "Insert the pink rod into the blue tube"


@configclass
class RC09PegInsertSceneCfg(InteractiveSceneCfg):
    """Tabletop scene: RC09 arm, pink rod, blue tube, cameras."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),
            rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    robot = RC09_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        spawn=sim_utils.CylinderCfg(
            radius=TUBE_RADIUS,
            height=TUBE_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.35, 0.95),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, TABLE_TOP_Z + TUBE_HEIGHT / 2),
        ),
    )

    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=PEG_RADIUS,
            height=PEG_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.45, 0.75),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.12, TABLE_TOP_Z + PEG_HEIGHT / 2),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + RC09_EE_BODY_NAME,
                name="ee",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.08)),
            ),
        ],
    )

    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.45),
            rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + RC09_EE_BODY_NAME + "/wrist_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.05),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros",
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=RC09_ARM_JOINTS,
        body_name=RC09_EE_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.08]),
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=RC09_GRIPPER_JOINTS,
        open_command_expr={name: 0.03 for name in RC09_GRIPPER_JOINTS},
        close_command_expr={name: 0.0 for name in RC09_GRIPPER_JOINTS},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
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
    insert_shaping = RewTerm(
        func=mdp.peg_insert_shaping,
        weight=0.15,
        params={"std": 0.06},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insert_success = DoneTerm(
        func=mdp.peg_insert_success,
        params={"xy_threshold": 0.008, "z_margin": 0.01},
    )


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_peg = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.40, 0.50),
                "y": (0.05, 0.20),
                "z": (TABLE_TOP_Z + PEG_HEIGHT / 2, TABLE_TOP_Z + PEG_HEIGHT / 2),
                "yaw": (-0.3, 0.3),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("peg"),
        },
    )


@configclass
class RC09PegInsertVisuomotorEnvCfg(ManagerBasedRLEnvCfg):
    scene: RC09PegInsertSceneCfg = RC09PegInsertSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    image_obs_list = ["table_cam", "wrist_cam"]

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.num_rerenders_on_reset = 2
