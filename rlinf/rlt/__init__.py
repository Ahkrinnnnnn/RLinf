# Copyright 2025 The RLinf Authors.
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

"""RLinf ↔ lerobot shared-core bridge for RLT Stage-2 (TD3 + BC).

This package is the RLinf-side glue that **reuses** lerobot's framework-agnostic
RLT core (networks, loss math, replay-window construction) instead of
re-implementing it. It deliberately contains no Ray worker mesh / FSDP / hydra
config — those concerns belong to RLinf's training backend. The two drivers
here are plain Python objects that can be:

* unit-tested in isolation (see ``tests/unit_tests/test_rlt_shared_core_conformance.py``),
* wrapped by RLinf workers (RolloutWorker / TrainingWorker) in a later phase,
* driven from lerobot's ``RLTStage2Orchestrator`` for real-robot deployment.

Because the math lives in ``lerobot.rlt.shared``, a checkpoint trained in RLinf
(simulation) loads bit-for-bit into lerobot (real robot) — enabling the
dual-framework cooperation described in ``docs/source/rlt_dual_framework.md``.
"""

from .chunked_rollout import ChunkedRolloutRunner, MockChunkEnv
from .td3_driver import RLTTD3Driver, RLTTD3DriverConfig

__all__ = ["RLTTD3Driver", "RLTTD3DriverConfig", "ChunkedRolloutRunner", "MockChunkEnv"]
