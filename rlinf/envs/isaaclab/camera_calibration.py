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

"""Load LeRobot hand-eye calibration sessions for Isaac Lab cameras.

Expected session layout (``lerobot-calibrate-hand-eye`` output)::

    {root}/{robot_id}/{experiment_name}/
        experiment_meta.json          # optional run index
        intrinsics/{camera}.json    # one file per camera
        phases/{phase_dir}/
            report.json               # extrinsics + phase metrics
            intrinsics.json           # optional phase-local copy

Extrinsics in ``report.json`` (LeRobot naming)::

    eye_in_hand : ``T_ee_to_camera``     with ``p_cam = T @ p_ee``
    eye_to_hand : ``T_camera_to_robot``  with ``p_robot = T @ p_cam``
                  (legacy alias: ``T_robot_to_camera``)

``CameraExtrinsicsSpec.transform_mm`` is always **camera → parent**
(``p_parent = R @ p_camera + t``, mm) so Isaac ``OffsetCfg`` can use it directly::

    from isaaclab.utils.math import quat_from_matrix

    cam_pos = T[:3, 3]                 # meters
    cam_rot = quat_from_matrix(T[:3, :3])
    OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros")

For eye-in-hand that means storing ``inv(T_ee_to_camera)``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

_EXTRINSIC_STD_KEYS = (
    "consistency_std_mm",
    "wrist_consistency_std_mm",
    "top_extrinsic_std_mm",
)
# Keys that indicate a report carries an extrinsic (any match is enough).
_EXTRINSIC_PRESENT_KEYS = (
    "T_ee_to_camera",
    "T_camera_to_robot",
    "T_robot_to_camera",  # legacy alias of T_camera_to_robot
)


class ExtrinsicPickStrategy(str, Enum):
    """How to choose among multiple phase reports for one camera."""

    BEST_STD = "best_std"
    LATEST = "latest"


@dataclass(frozen=True)
class HandEyeSessionMeta:
    robot_id: str
    robot_type: str
    experiment_name: str
    runs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class CameraIntrinsicsSpec:
    name: str
    width: int
    height: int
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    reprojection_error_px: float | None = None

    @property
    def intrinsic_matrix_row_major(self) -> list[float]:
        k = self.camera_matrix
        return [
            k[0][0],
            0.0,
            k[0][2],
            0.0,
            k[1][1],
            k[1][2],
            0.0,
            0.0,
            1.0,
        ]


@dataclass(frozen=True)
class CameraExtrinsicsSpec:
    """Camera→parent extrinsic for Isaac ``OffsetCfg``: ``p_parent = R @ p_camera + t``."""

    mount: Literal["eye_in_hand", "eye_to_hand"]
    parent_frame: str
    transform_mm: list[list[float]]
    report_path: Path
    quality_std_mm: float | None = None

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.transform_mm, dtype=np.float64)[:3, :3]

    @property
    def translation_mm(self) -> np.ndarray:
        return np.asarray(self.transform_mm, dtype=np.float64)[:3, 3]

    def offset_pos_m(self, length_unit: Literal["mm", "m"] = "mm") -> tuple[float, float, float]:
        scale = 1.0 if length_unit == "m" else 0.001
        t = self.translation_mm * scale
        return (float(t[0]), float(t[1]), float(t[2]))

    def offset_rot_wxyz_ros(self) -> tuple[float, float, float, float]:
        """Quaternion ``(w, x, y, z)`` from calib ``R`` via Isaac Lab ``quat_from_matrix``."""
        import torch
        from isaaclab.utils.math import quat_from_matrix

        mat = torch.as_tensor(self.rotation, dtype=torch.float32).unsqueeze(0)
        quat = quat_from_matrix(mat)[0]
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


@dataclass(frozen=True)
class CalibratedCameraSpec:
    camera_name: str
    intrinsics: CameraIntrinsicsSpec
    extrinsics: CameraExtrinsicsSpec


@dataclass(frozen=True)
class CameraSimBinding:
    """Map a calibration camera name to an Isaac Lab sensor + parent link."""

    camera_name: str
    parent_link: str


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_hand_eye_search_roots() -> list[Path]:
    """Search roots that contain ``{robot_id}/{experiment}/`` sessions."""
    roots: list[Path] = []
    for env_key in ("HAND_EYE_CALIB_ROOT", "RC09_CAMERA_CALIB_DIR"):
        if value := os.environ.get(env_key):
            roots.append(Path(value).expanduser().resolve())
    roots.append((_workspace_root().parent / "lerobot/outputs/models/hand_eye_calibration").resolve())
    return roots


def resolve_session_root(
    robot_id: str,
    experiment: str = "together",
    *,
    session_root: Path | str | None = None,
    search_roots: list[Path] | None = None,
) -> Path:
    """Resolve ``.../hand_eye_calibration/{robot_id}/{experiment}``."""
    if session_root is not None:
        path = Path(session_root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Calibration session directory not found: {path}")
        return path

    candidates = []
    for root in search_roots or default_hand_eye_search_roots():
        candidate = root / robot_id / experiment
        if candidate.is_dir():
            candidates.append(candidate)
        if root.name == experiment and root.parent.name == robot_id:
            candidates.append(root)

    if not candidates:
        searched = ", ".join(str(r / robot_id / experiment) for r in (search_roots or default_hand_eye_search_roots()))
        raise FileNotFoundError(
            f"No hand-eye session found for robot_id={robot_id!r} experiment={experiment!r}. "
            f"Searched: {searched}. Set HAND_EYE_CALIB_ROOT or pass session_root=."
        )
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _report_quality_std_mm(report: dict[str, Any]) -> float | None:
    values = [float(report[key]) for key in _EXTRINSIC_STD_KEYS if key in report]
    return min(values) if values else None


def _invert_transform_mm(transform: list[list[float]] | np.ndarray) -> list[list[float]]:
    """Invert a 4x4 rigid transform (mm-compatible translation column)."""
    t = np.asarray(transform, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    r = t[:3, :3]
    p = t[:3, 3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out.tolist()


def _parse_mount_and_transform(report: dict[str, Any]) -> tuple[str, list[list[float]]]:
    """Return ``(mount, T_camera_to_parent_mm)`` for Isaac ``OffsetCfg``."""
    mount = report.get("mount")

    if mount == "eye_in_hand" or (mount is None and "T_ee_to_camera" in report):
        if "T_ee_to_camera" not in report:
            raise KeyError("Report mount='eye_in_hand' but missing 'T_ee_to_camera'.")
        # T_ee_to_camera: p_cam = T @ p_ee.  Physical install (= README
        # camera_origin_in_ee_mm) is inv(T) translation — never use T[:3,3] as offset.
        return "eye_in_hand", _invert_transform_mm(report["T_ee_to_camera"])

    if mount == "eye_to_hand" or mount is None:
        for key in ("T_camera_to_robot", "T_robot_to_camera"):
            if key in report:
                return "eye_to_hand", report[key]
        if mount == "eye_to_hand":
            raise KeyError(
                "Report mount='eye_to_hand' but missing 'T_camera_to_robot' "
                "(or legacy 'T_robot_to_camera')."
            )

    raise KeyError(
        "Report has no T_ee_to_camera / T_camera_to_robot (or legacy T_robot_to_camera) extrinsic."
    )


def _parent_frame_for_mount(mount: str) -> str:
    # eye_in_hand parent is CRP/teach TCP (ee_frame target), not the bare URDF link.
    return "tcp" if mount == "eye_in_hand" else "robot"


class HandEyeCalibrationSession:
    """Reader for one ``{robot_id}/{experiment}`` calibration session."""

    def __init__(self, session_root: Path | str):
        self.session_root = Path(session_root).expanduser().resolve()
        if not self.session_root.is_dir():
            raise FileNotFoundError(f"Calibration session not found: {self.session_root}")
        self.meta = self._load_meta()

    @classmethod
    def from_robot(
        cls,
        robot_id: str,
        experiment: str = "together",
        *,
        session_root: Path | str | None = None,
        search_roots: list[Path] | None = None,
    ) -> HandEyeCalibrationSession:
        root = resolve_session_root(
            robot_id,
            experiment,
            session_root=session_root,
            search_roots=search_roots,
        )
        return cls(root)

    def _load_meta(self) -> HandEyeSessionMeta | None:
        meta_path = self.session_root / "experiment_meta.json"
        if not meta_path.is_file():
            return None
        data = _load_json(meta_path)
        return HandEyeSessionMeta(
            robot_id=str(data.get("robot_id", "")),
            robot_type=str(data.get("robot_type", "")),
            experiment_name=str(data.get("experiment_name", self.session_root.name)),
            runs=tuple(data.get("runs", [])),
        )

    @property
    def intrinsics_dir(self) -> Path:
        return self.session_root / "intrinsics"

    @property
    def phases_dir(self) -> Path:
        return self.session_root / "phases"

    def list_cameras(self) -> list[str]:
        if not self.intrinsics_dir.is_dir():
            raise FileNotFoundError(f"Missing intrinsics directory: {self.intrinsics_dir}")
        names = sorted(path.stem for path in self.intrinsics_dir.glob("*.json"))
        if not names:
            raise FileNotFoundError(f"No intrinsics JSON files in {self.intrinsics_dir}")
        return names

    def load_intrinsics(self, camera_name: str) -> CameraIntrinsicsSpec:
        path = self.intrinsics_dir / f"{camera_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Intrinsics not found for camera {camera_name!r}: {path}")
        data = _load_json(path)
        return CameraIntrinsicsSpec(
            name=camera_name,
            width=int(data["width"]),
            height=int(data["height"]),
            camera_matrix=data["camera_matrix"],
            dist_coeffs=data["dist_coeffs"],
            reprojection_error_px=data.get("reprojection_error_px"),
        )

    def _meta_run_candidates(
        self, camera_name: str
    ) -> list[tuple[float | None, str, Path]]:
        """Optional index from ``experiment_meta.json`` (best extrinsic run selection).

        ``experiment_meta.json`` is **not required**. When absent, extrinsics are
        discovered by scanning ``phases/*/report.json`` directly.
        """
        if self.meta is None:
            return []
        items: list[tuple[float | None, str, Path]] = []
        for run in self.meta.runs:
            if run.get("camera") != camera_name:
                continue
            rel_dir = run.get("dir")
            if not rel_dir:
                continue
            report_path = self.session_root / rel_dir / "report.json"
            if not report_path.is_file():
                continue
            std_values = [float(run[key]) for key in _EXTRINSIC_STD_KEYS if key in run]
            std = min(std_values) if std_values else None
            timestamp = str(run.get("timestamp_utc", report_path.parent.name))
            items.append((std, timestamp, report_path))
        return items

    def _candidate_report_paths(self, camera_name: str, phase: str | None) -> list[Path]:
        if phase is not None:
            for phase_dir in self.phases_dir.iterdir() if self.phases_dir.is_dir() else []:
                if phase in phase_dir.name:
                    report = phase_dir / "report.json"
                    if report.is_file():
                        return [report]
            raise FileNotFoundError(f"No phase report matching phase={phase!r} under {self.phases_dir}")

        paths: list[Path] = []
        seen: set[Path] = set()

        for _, _, report in self._meta_run_candidates(camera_name):
            if report not in seen:
                paths.append(report)
                seen.add(report)

        if self.phases_dir.is_dir():
            for report in sorted(self.phases_dir.glob("*/report.json")):
                if report in seen:
                    continue
                try:
                    data = _load_json(report)
                except json.JSONDecodeError:
                    continue
                if data.get("camera") not in (None, camera_name):
                    continue
                if _has_extrinsic(data):
                    paths.append(report)
                    seen.add(report)

        return paths

    def load_extrinsics(
        self,
        camera_name: str,
        *,
        phase: str | None = None,
        strategy: ExtrinsicPickStrategy | str = ExtrinsicPickStrategy.BEST_STD,
    ) -> CameraExtrinsicsSpec:
        strategy = ExtrinsicPickStrategy(strategy)

        if phase is not None:
            candidates = self._candidate_report_paths(camera_name, phase)
            scored: list[tuple[float, str, Path, dict[str, Any] | None]] = []
            for path in candidates:
                report = _load_json(path)
                if not _has_extrinsic(report):
                    continue
                std = _report_quality_std_mm(report)
                scored.append((std if std is not None else float("inf"), path.parent.name, path, report))
        else:
            meta_items = self._meta_run_candidates(camera_name)
            if meta_items:
                scored = [
                    (item[0] if item[0] is not None else float("inf"), item[1], item[2], None)
                    for item in meta_items
                ]
            else:
                candidates = self._candidate_report_paths(camera_name, phase=None)
                scored = []
                for path in candidates:
                    report = _load_json(path)
                    if not _has_extrinsic(report):
                        continue
                    std = _report_quality_std_mm(report)
                    scored.append((std if std is not None else float("inf"), path.parent.name, path, report))

        if not scored:
            raise FileNotFoundError(
                f"No extrinsic report found for camera {camera_name!r} under {self.phases_dir}"
            )

        if strategy == ExtrinsicPickStrategy.BEST_STD:
            scored.sort(key=lambda item: (item[0], item[1]))
        else:
            scored.sort(key=lambda item: item[1])

        quality_std, _, path, cached_report = scored[0]
        report = cached_report if cached_report is not None else _load_json(path)
        mount, transform = _parse_mount_and_transform(report)
        return CameraExtrinsicsSpec(
            mount=mount,
            parent_frame=_parent_frame_for_mount(mount),
            transform_mm=transform,
            report_path=path,
            quality_std_mm=quality_std if quality_std != float("inf") else _report_quality_std_mm(report),
        )

    def load_camera(
        self,
        camera_name: str,
        *,
        phase: str | None = None,
        strategy: ExtrinsicPickStrategy | str = ExtrinsicPickStrategy.BEST_STD,
        require_extrinsics: bool = True,
    ) -> CalibratedCameraSpec:
        intrinsics = self.load_intrinsics(camera_name)
        extrinsics = None
        if require_extrinsics:
            extrinsics = self.load_extrinsics(camera_name, phase=phase, strategy=strategy)
        if extrinsics is None:
            raise ValueError(f"Extrinsics required but not loaded for camera {camera_name!r}.")
        return CalibratedCameraSpec(
            camera_name=camera_name,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    def load_for_sim(
        self,
        bindings: dict[str, CameraSimBinding],
        *,
        strategy: ExtrinsicPickStrategy | str = ExtrinsicPickStrategy.BEST_STD,
    ) -> dict[str, tuple[CalibratedCameraSpec, CameraSimBinding]]:
        """Load cameras and pair each with its sim binding (sensor name → spec)."""
        loaded: dict[str, tuple[CalibratedCameraSpec, CameraSimBinding]] = {}
        for sensor_name, binding in bindings.items():
            spec = self.load_camera(binding.camera_name, strategy=strategy)
            loaded[sensor_name] = (spec, binding)
        return loaded


def _has_extrinsic(report: dict[str, Any]) -> bool:
    return any(key in report for key in _EXTRINSIC_PRESENT_KEYS)
