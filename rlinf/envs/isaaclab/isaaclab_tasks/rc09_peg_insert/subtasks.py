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

"""RC09 tabletop subtasks — sequential assembly, different start scenes.

Pipeline (each skill assumes prior steps are done):

1. ``stack_base``  — purple middle is **beside** empty black base → stack into base
2. ``insert_tube`` — purple is **already nested** in black base → insert blue tube
3. ``insert_rod``  — blue tube is **already seated** in purple → insert pink rod

Active subtask: ``RC09_SUBTASK`` (default ``insert_tube``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal


_DEG = math.pi / 180.0

# Where each movable CAD part starts for this skill.
MiddlePose = Literal["beside_base", "nested_in_base"]
TubePose = Literal["on_table", "seated_in_purple"]
PegPose = Literal["on_table", "seated_in_tube"]


@dataclass(frozen=True)
class RC09SubtaskSpec:
    """One language skill + start-scene layout + dataset-aligned home."""

    id: str
    prompt: str
    dataset_dirname: str
    # First-frame mean of ``observation.state`` (deg), j1…j6.
    home_joints_deg: tuple[float, float, float, float, float, float]
    # Start layout (prior assembly already applied).
    middle_pose: MiddlePose
    tube_pose: TubePose
    peg_pose: PegPose
    # Which parts the policy may grasp / move this skill.
    peg_dynamic: bool
    tube_dynamic: bool
    middle_dynamic: bool

    @property
    def middle_nested_in_base(self) -> bool:
        return self.middle_pose == "nested_in_base"


# Homes ≈ first-frame means over the first ~20 episodes of each LeRobot set.
RC09_SUBTASKS: dict[str, RC09SubtaskSpec] = {
    # Skill 1: assemble purple into black. Tube/peg are side clutter only.
    "stack_base": RC09SubtaskSpec(
        id="stack_base",
        prompt="Stack the purple base on the black base",
        dataset_dirname="stack_the_base_all",
        home_joints_deg=(-17.08, 70.61, -37.06, 68.69, 2.72, -19.99),
        middle_pose="beside_base",
        tube_pose="on_table",
        peg_pose="on_table",
        peg_dynamic=False,
        tube_dynamic=False,
        middle_dynamic=True,
    ),
    # Skill 2: purple already in black; pick blue tube and insert into purple.
    "insert_tube": RC09SubtaskSpec(
        id="insert_tube",
        prompt="Insert the blue tube into the purple base",
        dataset_dirname="insert_the_blue_tube_all",
        home_joints_deg=(-8.96, 81.21, -39.33, 48.76, 1.16, -9.68),
        middle_pose="nested_in_base",
        tube_pose="on_table",
        peg_pose="on_table",
        peg_dynamic=False,
        tube_dynamic=True,
        middle_dynamic=False,
    ),
    # Skill 3: tube already seated in purple/base stack; insert pink rod into tube.
    "insert_rod": RC09SubtaskSpec(
        id="insert_rod",
        prompt="Insert the pink rod into the blue tube",
        dataset_dirname="insert_the_rod_all",
        home_joints_deg=(-8.47, 75.81, -27.67, 42.52, 1.15, -9.19),
        middle_pose="nested_in_base",
        tube_pose="seated_in_purple",
        peg_pose="on_table",
        peg_dynamic=True,
        tube_dynamic=False,
        middle_dynamic=False,
    ),
}


def resolve_active_subtask(subtask_id: str | None = None) -> RC09SubtaskSpec:
    """Resolve subtask from arg or ``RC09_SUBTASK`` (default ``insert_tube``)."""
    key = (subtask_id or os.environ.get("RC09_SUBTASK") or "insert_tube").strip()
    if key not in RC09_SUBTASKS:
        known = ", ".join(sorted(RC09_SUBTASKS))
        raise KeyError(f"Unknown RC09 subtask {key!r}. Expected one of: {known}")
    return RC09_SUBTASKS[key]


def home_joint_pos_rad(
    spec: RC09SubtaskSpec,
    *,
    joint_signs: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Build ``RC09_DEFAULT_ARM_POS`` arm entries (rad) from a subtask home."""
    names = ("J1_Joint", "J2_Joint", "J3_joint", "J4_Joint", "J5_Joint", "J6_Joint")
    return {
        name: deg * _DEG * sign
        for name, deg, sign in zip(names, spec.home_joints_deg, joint_signs, strict=True)
    }


def default_dataset_root(spec: RC09SubtaskSpec) -> str:
    """Default LeRobot root on the lab external drive."""
    return f"/media/lenovo/My Passport/{spec.dataset_dirname}"
