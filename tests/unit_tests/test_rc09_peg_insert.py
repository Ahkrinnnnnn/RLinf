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

"""Unit tests for RC09 peg-insert Isaac Lab integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBODIED_CONFIG = REPO_ROOT / "examples" / "embodiment" / "config"


class TestEnvRegistration:
    def test_peg_insert_task_registered(self):
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
        from rlinf.envs.isaaclab.tasks.rc09_peg_insert import IsaaclabRC09PegInsertEnv

        assert "Isaac-RC09-PegInsert-Visuomotor-v0" in REGISTER_ISAACLAB_ENVS
        assert REGISTER_ISAACLAB_ENVS["Isaac-RC09-PegInsert-Visuomotor-v0"] is IsaaclabRC09PegInsertEnv

    def test_reach_task_removed(self):
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        assert "Isaac-RC09-Reach-Visuomotor-v0" not in REGISTER_ISAACLAB_ENVS


class TestOpenPIPolicyIO:
    def test_inputs_outputs_roundtrip(self):
        from openpi.models import model as _model

        from rlinf.models.embodiment.openpi.policies.isaaclab_rc09_policy import (
            IsaacLabRC09Inputs,
            IsaacLabRC09Outputs,
            make_isaaclab_rc09_example,
        )

        example = make_isaaclab_rc09_example()
        inputs = IsaacLabRC09Inputs(model_type=_model.ModelType.PI05)(example)
        assert "state" in inputs
        assert inputs["state"].shape[-1] == 6
        assert "image" in inputs
        assert inputs["image"]["base_0_rgb"].shape[:2] == (480, 640)

        fake_out = {"actions": np.random.randn(5, 7).astype(np.float32)}
        actions = IsaacLabRC09Outputs()(fake_out)["actions"]
        assert actions.shape == (5, 7)

    def test_real_robot_unit_conversion(self):
        import math

        from rlinf.envs.isaaclab.rc09_real_robot_io import (
            clip_policy_action_to_limits,
            policy_action_to_sim,
            policy_arm_joints_to_sim,
            sim_joint_pos_to_policy,
        )

        rad = torch.tensor([[0.0, math.pi / 2, -1.0, 0.5, 0.1, -0.2]])
        deg = sim_joint_pos_to_policy(rad)
        assert torch.allclose(deg, torch.tensor([[0.0, 90.0, -57.2958, 28.6479, 5.7296, -11.4592]]), atol=1e-3)

        policy_act = torch.tensor([[0.0, 90.0, -36.0, 37.0, 1.0, -6.0, 800.0]])
        sim_act = policy_action_to_sim(policy_act)
        assert torch.allclose(
            sim_act[0, :6],
            policy_arm_joints_to_sim(policy_act[0, :6]),
            atol=1e-4,
        )
        assert sim_act[0, 6].item() == 1.0

        from rlinf.envs.isaaclab.rc09_real_robot_io import clip_policy_action_to_limits

        closed = policy_action_to_sim(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]]))
        assert closed[0, 6].item() == -1.0

        over = clip_policy_action_to_limits(
            torch.tensor([[0.0, 400.0, 300.0, 0.0, 0.0, 0.0, 2000.0]])
        )
        assert over[0, 1].item() == 360.0
        assert over[0, 2].item() == 256.0
        assert over[0, 6].item() == 1000.0

    def test_dataconfig_registered(self):
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        cfg = get_openpi_config("pi05_isaaclab_rc09_peg_insert")
        assert cfg.name == "pi05_isaaclab_rc09_peg_insert"
        assert "pink rod" in cfg.data.default_prompt.lower()


class TestRLTokenBridge:
    def test_augment_critic_disabled(self):
        from rlinf.models.embodiment.openpi.rlt_bridge import (
            RLTokenBridge,
            RLTokenBridgeConfig,
        )

        bridge = RLTokenBridge(RLTokenBridgeConfig(enabled=False))
        critic = torch.randn(4, 2048)
        out = bridge.augment_critic_input(critic, None)
        assert torch.equal(out, critic)

    def test_augment_critic_concat(self):
        from rlinf.models.embodiment.openpi.rlt_bridge import (
            RLTokenBridge,
            RLTokenBridgeConfig,
        )

        bridge = RLTokenBridge(RLTokenBridgeConfig(enabled=True, checkpoint_path="/tmp/x"))
        critic = torch.randn(4, 2048)
        rl_token = torch.randn(4, 1, 2048)
        out = bridge.augment_critic_input(critic, rl_token)
        assert out.shape == (4, 4096)


class TestHydraConfigs:
    @pytest.fixture
    def config_dir(self):
        return EMBODIED_CONFIG

    def test_eval_yaml_structure(self):
        cfg = OmegaConf.load(REPO_ROOT / "evaluations/isaaclab/rc09_peg_insert_pi05_eval.yaml")
        assert cfg.runner.task_type == "embodied_eval"
        assert cfg.runner.only_eval is True
        assert cfg.env.eval.total_num_envs == 32
        assert cfg.env.eval.ignore_terminations is True
        assert cfg.rollout.model.openpi.config_name == "pi05_isaaclab_rc09_peg_insert"
        assert cfg.rollout.model.action_dim == 7

    def test_env_yaml_task_id(self, config_dir):
        cfg = OmegaConf.load(config_dir / "rc09_peg_insert/env.yaml")
        assert cfg.init_params.id == "Isaac-RC09-PegInsert-Visuomotor-v0"
        assert "pink rod" in cfg.init_params.task_description.lower()

    def test_ppo_pi05_yaml_structure(self, config_dir):
        cfg = OmegaConf.load(config_dir / "rc09_peg_insert/ppo_pi05.yaml")
        assert cfg.actor.model.model_type == "openpi"
        assert cfg.actor.model.openpi.config_name == "pi05_isaaclab_rc09_peg_insert"
        assert cfg.actor.model.openpi.use_rl_token is False

    def test_ppo_pi05_rlt_enables_token(self, config_dir):
        cfg = OmegaConf.load(config_dir / "rc09_peg_insert/ppo_pi05_rlt.yaml")
        assert cfg.actor.model.openpi.use_rl_token is True
        assert cfg.actor.model.openpi.rl_token_checkpoint is not None


class TestRC09GripperForceTermination:
    def test_excessive_force_on_either_finger(self):
        import torch

        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert import mdp

        class FakeSensor:
            def __init__(self, force_norm):
                self.data = type(
                    "Data",
                    (),
                    {
                        "net_forces_w_history": torch.tensor([[[[0.0, force_norm, 0.0]]]]),
                        "net_forces_w": None,
                    },
                )()

        class FakeScene:
            def __init__(self, left_norm, right_norm):
                self._left = FakeSensor(left_norm)
                self._right = FakeSensor(right_norm)

            def __getitem__(self, name):
                if name == "contact_force_left":
                    return self._left
                if name == "contact_force_right":
                    return self._right
                raise KeyError(name)

        class FakeEnv:
            def __init__(self, left_norm, right_norm):
                self.scene = FakeScene(left_norm, right_norm)

        term = mdp.terminations.gripper_excessive_force(FakeEnv(10.0, 50.0), force_threshold=40.0)
        assert term.tolist() == [True]

        term = mdp.terminations.gripper_excessive_force(FakeEnv(10.0, 10.0), force_threshold=40.0)
        assert term.tolist() == [False]


class TestRC09Terminations:
    def test_consecutive_success_requires_multiple_steps(self):
        import torch

        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert import mdp

        class FakeEnv:
            num_envs = 2
            device = torch.device("cpu")

        env = FakeEnv()

        def fake_success(*args, **kwargs):
            # env 0 always success, env 1 alternates
            return torch.tensor([1.0, 0.0])

        original = mdp.rewards.peg_insert_success
        mdp.rewards.peg_insert_success = fake_success
        try:
            term = mdp.terminations.peg_insert_success_consecutive(env, num_steps=3)
            assert term.tolist() == [False, False]
            term = mdp.terminations.peg_insert_success_consecutive(env, num_steps=3)
            assert term.tolist() == [False, False]
            term = mdp.terminations.peg_insert_success_consecutive(env, num_steps=3)
            assert term.tolist() == [True, False]

            mdp.terminations.reset_success_streak(env, torch.tensor([0]))
            assert env._rc09_success_streak.tolist() == [0, 2]
        finally:
            mdp.rewards.peg_insert_success = original


class TestHandEyeCameraCalibration:
    def test_session_lists_cameras(self):
        from rlinf.envs.isaaclab.camera_calibration import (
            HandEyeCalibrationSession,
            resolve_session_root,
        )

        root = resolve_session_root("lab_arm_01", "together")
        session = HandEyeCalibrationSession(root)
        assert session.list_cameras() == ["top", "wrist"]

    def test_load_rc09_calibrated_cameras(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.camera_calibration import (
            load_rc09_calibrated_cameras,
        )

        cams = load_rc09_calibrated_cameras()
        top_spec, top_bind = cams["table_cam"]
        wrist_spec, wrist_bind = cams["wrist_cam"]

        assert top_bind.camera_name == "top" and top_bind.parent_link == "base_link"
        assert wrist_bind.camera_name == "wrist" and wrist_bind.parent_link == "PGC_300_60"

        assert top_spec.intrinsics.width == 640 and top_spec.intrinsics.height == 480
        assert top_spec.extrinsics.mount == "eye_to_hand"
        assert wrist_spec.extrinsics.mount == "eye_in_hand"
        assert wrist_spec.extrinsics.parent_frame == "tcp"
        assert top_spec.intrinsics.reprojection_error_px is not None
        assert top_spec.intrinsics.reprojection_error_px < 1.0

        top_pos = top_spec.extrinsics.offset_pos_m()
        # Current lab mount: ~ (0.04, -0.38, 0.22) m in robot base.
        assert -0.2 < top_pos[0] < 0.2
        assert -0.6 < top_pos[1] < -0.2
        assert 0.1 < top_pos[2] < 0.4
        # Pose must match calib report exactly (mm→m, no extra R flips).
        assert top_pos == pytest.approx(
            (
                top_spec.extrinsics.translation_mm[0] * 0.001,
                top_spec.extrinsics.translation_mm[1] * 0.001,
                top_spec.extrinsics.translation_mm[2] * 0.001,
            )
        )
        assert len(top_spec.intrinsics.intrinsic_matrix_row_major) == 9

        # Wrist loaded transform is camera→TCP (= camera_origin_in_ee_mm), not T_ee_to_camera[:3,3].
        wrist_pos = wrist_spec.extrinsics.offset_pos_m()
        wrist_dist = (wrist_pos[0] ** 2 + wrist_pos[1] ** 2 + wrist_pos[2] ** 2) ** 0.5
        assert 0.05 < wrist_dist < 0.35
        assert wrist_spec.extrinsics.quality_std_mm is not None
        assert wrist_spec.extrinsics.quality_std_mm < 10.0
        # Must NOT equal raw T_ee_to_camera translation (~0.01, 0.03, 0.18).
        assert abs(wrist_pos[1]) > 0.05  # camera_origin y ~ +0.10 m

    def test_picks_best_extrinsic_std_from_meta(self):
        from rlinf.envs.isaaclab.camera_calibration import (
            ExtrinsicPickStrategy,
            HandEyeCalibrationSession,
            resolve_session_root,
        )

        session = HandEyeCalibrationSession(resolve_session_root("lab_arm_01", "together"))
        ext = session.load_extrinsics("top", strategy=ExtrinsicPickStrategy.BEST_STD)
        assert ext.quality_std_mm is not None
        assert ext.quality_std_mm < 10.0


class TestRC09DomainRandomization:
    def test_domain_randomization_enabled_default(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert import mdp

        class FakeCfg:
            enable_domain_randomization = True

        class FakeEnv:
            cfg = FakeCfg()

        assert mdp.domain_randomization_enabled(FakeEnv()) is True

    def test_domain_randomization_disabled(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert import mdp

        class FakeCfg:
            enable_domain_randomization = False

        class FakeEnv:
            cfg = FakeCfg()

        assert mdp.domain_randomization_enabled(FakeEnv()) is False

    def test_sample_random_color_in_unit_range(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.mdp.events import sample_random_color

        for _ in range(20):
            color = sample_random_color(base=(0.75, 0.75, 0.75), variation=0.4)
            assert len(color) == 3
            assert all(0.0 <= channel <= 1.0 for channel in color)

    def test_event_cfg_includes_dr_terms(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.rc09_peg_insert_env_cfg import EventCfg

        assert hasattr(EventCfg, "randomize_light")
        assert hasattr(EventCfg, "randomize_peg_color")
        assert hasattr(EventCfg, "peg_mass")

    def test_peg_tube_reset_uses_relative_table_offsets(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.rc09_peg_insert_env_cfg import (
            TABLE_CENTER_XY,
            TABLE_SIZE,
            TABLE_TOP_Z,
            EventCfgBase,
            PEG_HEIGHT,
            PEG_INIT_XY,
            RC09PegInsertSceneCfg,
            TUBE_HEIGHT,
            TUBE_INIT_XY,
        )

        # Table length (Y) is 1.6 m; depth (X) is the shorter 0.424 m edge.
        assert TABLE_SIZE[0] == pytest.approx(0.424)
        assert TABLE_SIZE[1] == pytest.approx(1.60)
        table_x_min = TABLE_CENTER_XY[0] - TABLE_SIZE[0] / 2
        table_x_max = TABLE_CENTER_XY[0] + TABLE_SIZE[0] / 2
        assert table_x_min < PEG_INIT_XY[0] < table_x_max
        assert table_x_min < TUBE_INIT_XY[0] < table_x_max
        events = EventCfgBase()
        assert events.reset_peg.params["pose_range"]["z"] == (0.0, 0.0)
        assert events.reset_tube.params["pose_range"]["z"] == (0.0, 0.0)
        assert events.reset_base.params["asset_cfg"].name == "base"
        assert events.reset_middle.params["asset_cfg"].name == "middle"
        assert events.reset_cover.params["asset_cfg"].name == "cover"

        scene = RC09PegInsertSceneCfg(num_envs=1, env_spacing=2.5)
        assert scene.peg.init_state.pos[2] == pytest.approx(TABLE_TOP_Z + PEG_HEIGHT / 2)
        assert scene.tube.init_state.pos[2] == pytest.approx(TABLE_TOP_Z + TUBE_HEIGHT / 2)
        assert scene.base.init_state.pos[:2] == scene.middle.init_state.pos[:2]
        assert scene.middle.init_state.pos[2] > scene.base.init_state.pos[2]
        assert scene.base.spawn.usd_path.endswith("base.usda")
        assert scene.middle.spawn.usd_path.endswith("middle.usda")
        assert scene.cover.spawn.usd_path.endswith("cover.usda")
        assert scene.peg.spawn.usd_path.endswith("peg.usda")
        assert scene.tube.spawn.usd_path.endswith("tube.usda")

    def test_env_cfg_disables_dr_events_when_flag_off(self):
        from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.rc09_peg_insert_env_cfg import (
            EventCfgBase,
            RC09PegInsertVisuomotorEnvCfg,
        )

        cfg = RC09PegInsertVisuomotorEnvCfg(enable_domain_randomization=False)
        cfg.__post_init__()
        assert isinstance(cfg.events, EventCfgBase)
        assert not hasattr(cfg.events, "randomize_light")

    def test_env_yaml_dr_flag(self, config_dir=None):
        cfg = OmegaConf.load(EMBODIED_CONFIG / "rc09_peg_insert/env.yaml")
        assert cfg.init_params.enable_domain_randomization is True

    def test_eval_yaml_disables_dr(self):
        cfg = OmegaConf.load(REPO_ROOT / "evaluations/isaaclab/rc09_peg_insert_pi05_eval.yaml")
        assert cfg.env.eval.init_params.enable_domain_randomization is False


class TestInstallScript:
    def test_install_script_uses_symlink(self):
        script = (
            REPO_ROOT
            / "rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh"
        )
        assert script.is_file()
        content = script.read_text(encoding="utf-8")
        assert "ln -s" in content
        assert "cp -r" not in content
        assert "Isaac-RC09-PegInsert-Visuomotor-v0" in content

    def test_rlinf_root_depth_in_install_script(self):
        """Install script must resolve RLinf root (6 levels up from scripts/)."""
        script = (
            REPO_ROOT
            / "rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh"
        )
        content = script.read_text(encoding="utf-8")
        assert 'realpath "${SCRIPT_DIR}/../../../../../../")' in content
