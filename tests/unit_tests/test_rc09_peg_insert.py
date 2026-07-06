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
        assert "image" in inputs
        assert inputs["image"]["base_0_rgb"].shape == (256, 256, 3)

        fake_out = {"actions": np.random.randn(5, 7).astype(np.float32)}
        actions = IsaacLabRC09Outputs()(fake_out)["actions"]
        assert actions.shape == (5, 7)
        assert np.all(np.isin(actions[..., -1], [-1.0, 1.0]))

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
