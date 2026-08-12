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

"""RC09 sim camera bindings on ``hand_eye_calibration/{robot}/{experiment}``.

Loads **JSON** from the LeRobot session (``intrinsics/*.json`` + ``phases/*/report.json``).
``calibration.npz`` also stores I/O, but the session JSON is preferred (std / multi-run pick).

Wrist (eye-in-hand):
  Calib stores ``T_ee_to_camera`` (``p_cam = T @ p_ee``). Isaac offset uses
  ``inv(T)`` whose translation is ``camera_origin_in_ee_mm`` (TCP→camera install),
  **not** the ``T`` translation column. Parent link is ``PGC_300_60``; TCP is that
  link plus ``RC09_TCP_OFFSET_POS_M`` (same as ``ee_frame`` target).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rlinf.envs.isaaclab.camera_calibration import (
    CalibratedCameraSpec,
    CameraSimBinding,
    ExtrinsicPickStrategy,
    HandEyeCalibrationSession,
)

if TYPE_CHECKING:
    from isaaclab.sensors import CameraCfg

RC09_CALIB_ROBOT_ID = "lab_arm_01"
RC09_CALIB_EXPERIMENT = "together"

# Keep here (no IsaacLab import) so unit tests can load calib without carb.
RC09_EE_BODY_NAME = "PGC_300_60"
# Approximate flange→TCP along EE Z (not pendant tool offset). Used only for
# ee_frame / wrist-cam composition; joint-position control ignores this.
RC09_TCP_OFFSET_POS_M: tuple[float, float, float] = (0.0, 0.0, 0.2)

# Parent links must match USD prim names after URDF import ("-" → "_").
# Wrist extrinsic is camera→TCP; TCP = EE link + RC09_TCP_OFFSET_POS_M.
RC09_CAMERA_BINDINGS: dict[str, CameraSimBinding] = {
    "table_cam": CameraSimBinding(camera_name="top", parent_link="base_link"),
    "wrist_cam": CameraSimBinding(camera_name="wrist", parent_link=RC09_EE_BODY_NAME),
}


def load_rc09_calibrated_cameras(
    *,
    robot_id: str = RC09_CALIB_ROBOT_ID,
    experiment: str = RC09_CALIB_EXPERIMENT,
    session_root: Path | str | None = None,
    strategy: ExtrinsicPickStrategy | str = ExtrinsicPickStrategy.BEST_STD,
) -> dict[str, tuple[CalibratedCameraSpec, CameraSimBinding]]:
    """Load top/wrist from ``lerobot/.../hand_eye_calibration`` JSON session."""
    session = HandEyeCalibrationSession.from_robot(
        robot_id,
        experiment,
        session_root=session_root,
    )
    return session.load_for_sim(RC09_CAMERA_BINDINGS, strategy=strategy)


def make_camera_cfg(
    spec: CalibratedCameraSpec,
    binding: CameraSimBinding,
    *,
    prim_path: str,
    clipping_range: tuple[float, float] = (0.1, 2.5),
    tcp_offset_in_link_m: tuple[float, float, float] = RC09_TCP_OFFSET_POS_M,
) -> CameraCfg:
    """Build Isaac ``CameraCfg`` (ROS convention, meters).

    For wrist: ``spec.extrinsics`` is camera→TCP (``camera_origin_in_ee``); compose
    with ``tcp_offset_in_link_m`` so the prim can hang under the EE USD link.
    """
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg as _CameraCfg
    from isaaclab.utils.math import quat_from_matrix

    w, h = spec.intrinsics.width, spec.intrinsics.height
    R_cam = np.asarray(spec.extrinsics.rotation, dtype=np.float64)
    t_cam = np.asarray(spec.extrinsics.offset_pos_m(), dtype=np.float64)

    if spec.extrinsics.mount == "eye_in_hand":
        # T_link_to_cam = T_link_to_tcp @ T_tcp_to_cam  (TCP rot = I in link)
        t_tcp = np.asarray(tcp_offset_in_link_m, dtype=np.float64)
        R_link = R_cam
        t_link = t_cam + t_tcp
    else:
        R_link = R_cam
        t_link = t_cam

    cam_pos = (float(t_link[0]), float(t_link[1]), float(t_link[2]))
    cam_rot = quat_from_matrix(torch.as_tensor(R_link, dtype=torch.float32).unsqueeze(0))[0]
    return _CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=w,
        height=h,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=spec.intrinsics.intrinsic_matrix_row_major,
            width=w,
            height=h,
            clipping_range=clipping_range,
        ),
        offset=_CameraCfg.OffsetCfg(
            pos=cam_pos,
            rot=(float(cam_rot[0]), float(cam_rot[1]), float(cam_rot[2]), float(cam_rot[3])),
            convention="ros",
        ),
    )
