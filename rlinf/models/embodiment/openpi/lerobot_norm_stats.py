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

"""Convert LeRobot policy preprocessor stats to OpenPI norm_stats.json."""

from __future__ import annotations

import glob
import pathlib

import numpy as np
import safetensors.torch
from openpi.shared import normalize as openpi_normalize


def _find_lerobot_normalizer_path(checkpoint_dir: pathlib.Path) -> pathlib.Path | None:
    patterns = [
        "policy_preprocessor_step_*_normalizer_processor.safetensors",
        "*normalizer*.safetensors",
    ]
    for pattern in patterns:
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _stats_from_prefix(tensors: dict[str, np.ndarray], prefix: str) -> openpi_normalize.NormStats:
    mean = np.asarray(tensors[f"{prefix}.mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(tensors[f"{prefix}.std"], dtype=np.float64).reshape(-1)
    q01 = np.asarray(tensors[f"{prefix}.q01"], dtype=np.float64).reshape(-1)
    q99 = np.asarray(tensors[f"{prefix}.q99"], dtype=np.float64).reshape(-1)
    return openpi_normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def load_norm_stats_from_lerobot_checkpoint(
    checkpoint_dir: pathlib.Path,
) -> dict[str, openpi_normalize.NormStats]:
    """Build OpenPI norm stats from a LeRobot ``pretrained_model`` directory."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    normalizer_path = _find_lerobot_normalizer_path(checkpoint_dir)
    if normalizer_path is None:
        raise FileNotFoundError(
            f"No LeRobot normalizer safetensors found under {checkpoint_dir}"
        )

    raw = safetensors.torch.load_file(str(normalizer_path), device="cpu")
    tensors = {key: value.numpy() for key, value in raw.items()}

    prefixes = {
        "state": "observation.state",
        "actions": "action",
    }
    norm_stats: dict[str, openpi_normalize.NormStats] = {}
    for openpi_key, lerobot_prefix in prefixes.items():
        if f"{lerobot_prefix}.mean" not in tensors:
            raise FileNotFoundError(
                f"Missing {lerobot_prefix} stats in {normalizer_path.name}"
            )
        norm_stats[openpi_key] = _stats_from_prefix(tensors, lerobot_prefix)
    return norm_stats


def write_norm_stats_for_asset(
    checkpoint_dir: pathlib.Path,
    asset_id: str,
    norm_stats: dict[str, openpi_normalize.NormStats],
) -> pathlib.Path:
    """Persist norm stats where OpenPI expects them inside a checkpoint."""
    output_dir = checkpoint_dir / asset_id
    openpi_normalize.save(output_dir, norm_stats)
    return output_dir / "norm_stats.json"


def maybe_load_norm_stats(
    checkpoint_dir: pathlib.Path,
    asset_id: str | None,
    norm_stats_path: str | None,
) -> dict[str, openpi_normalize.NormStats]:
    """Load OpenPI norm stats, falling back to LeRobot preprocessor stats."""
    from openpi.training import checkpoints as openpi_checkpoints

    if norm_stats_path is not None:
        norm_dir = pathlib.Path(norm_stats_path).expanduser()
        if norm_dir.is_file():
            norm_dir = norm_dir.parent
        return openpi_checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)

    if asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")

    try:
        return openpi_checkpoints.load_norm_stats(checkpoint_dir, asset_id)
    except FileNotFoundError:
        norm_stats = load_norm_stats_from_lerobot_checkpoint(checkpoint_dir)
        write_norm_stats_for_asset(checkpoint_dir, asset_id, norm_stats)
        return norm_stats
