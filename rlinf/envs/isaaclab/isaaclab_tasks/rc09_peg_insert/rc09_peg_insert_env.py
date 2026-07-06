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

"""Thin env subclass hook for RC09 peg-insert (logic lives in env cfg)."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .rc09_peg_insert_env_cfg import RC09PegInsertVisuomotorEnvCfg


class RC09PegInsertVisuomotorEnv(ManagerBasedRLEnv):
    """RC09 peg-insert visuomotor environment."""

    cfg: RC09PegInsertVisuomotorEnvCfg
