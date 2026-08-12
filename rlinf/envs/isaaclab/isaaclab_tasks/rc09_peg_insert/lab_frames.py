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

"""Archived CRP pendant frames (mm / deg → SI) — **not wired into the sim**.

Joint-position datasets (``j1.pos``…``j6.pos``) are independent of the user
frame. Tool offset only matters for TCP / EE-based object inference.

Active sim uses ``camera_calibration.RC09_TCP_OFFSET_POS_M = (0, 0, 0.2)`` and
``TABLE_CENTER_XY = (0.50, 0.0)`` in ``rc09_peg_insert_env_cfg`` instead.
Keep these numbers here only as lab reference if Cartesian alignment is needed
later.
"""

from __future__ import annotations

import math

import numpy as np

# User frame relative to CRP world / base (as reported on pendant).
RC09_USER_POS_M: tuple[float, float, float] = (
    457.74 * 0.001,
    -107.27 * 0.001,
    -385.55 * 0.001,
)
RC09_USER_RPY_DEG: tuple[float, float, float] = (179.70, -0.37, -3.30)

# Tool offset: flange/PGC link → TCP (pendant "工具偏移") — unused in active cfg.
RC09_TOOL_POS_M: tuple[float, float, float] = (
    -50.133 * 0.001,
    10.875 * 0.001,
    233.464 * 0.001,
)
RC09_TOOL_RPY_DEG: tuple[float, float, float] = (-1.386, -0.379, 94.022)


def rpy_deg_to_rot_matrix(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    """Fixed-frame XYZ R = Rz @ Ry @ Rx (common industrial convention)."""
    rx, ry, rz = (math.radians(v) for v in rpy_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def tcp_offset_rot_wxyz() -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for pendant tool orientation (reference only)."""
    r = rpy_deg_to_rot_matrix(RC09_TOOL_RPY_DEG)
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def user_origin_world_m(*, robot_base_z: float) -> tuple[float, float, float]:
    """User-frame origin in sim world if robot base is at (0,0,robot_base_z)."""
    return (
        RC09_USER_POS_M[0],
        RC09_USER_POS_M[1],
        robot_base_z + RC09_USER_POS_M[2],
    )
