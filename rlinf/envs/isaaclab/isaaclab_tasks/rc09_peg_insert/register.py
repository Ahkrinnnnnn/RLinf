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

"""Gymnasium registration for RC09 peg-insert tasks."""

from __future__ import annotations

import gymnasium as gym


def register_rc09_peg_insert_envs() -> None:
    """Register RC09 peg-insert environments with Gymnasium."""
    module = "isaaclab_tasks.manager_based.manipulation.rc09_peg_insert"

    gym.register(
        id="Isaac-RC09-PegInsert-Visuomotor-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{module}.rc09_peg_insert_env_cfg:RC09PegInsertVisuomotorEnvCfg"
            ),
        },
    )
