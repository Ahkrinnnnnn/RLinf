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

"""Utils for evaluating policies in Issaaclab simulation environments."""

import pickle
import sys

import cloudpickle
import torch

# Ray worker processes inject these into sys.argv. SimulationApp/AppLauncher
# forwards unknown argv into Kit, which breaks GUI viewport startup.
_RAY_KIT_ARGV_PREFIXES = (
    "--node-ip-address",
    "--node-manager-port",
    "--object-store-name",
    "--raylet-name",
    "--redis-address",
    "--metrics-agent-port",
    "--logging-rotate-bytes",
    "--logging-rotate-backup-count",
    "--runtime-env-agent-port",
    "--gcs-address",
    "--session-name",
    "--temp-dir",
    "--webui",
    "--cluster-id",
    "--startup-token",
    "--worker-launch-time-ms",
    "--node-id",
    "--runtime-env-hash",
)


def scrub_sys_argv_for_isaac_kit() -> list[str]:
    """Remove Ray (and similar) CLI args so AppLauncher does not pass them to Kit.

    Returns the removed arguments (for logging).
    """
    if not sys.argv:
        return []
    kept = [sys.argv[0]]
    removed: list[str] = []
    for arg in sys.argv[1:]:
        if any(arg == p or arg.startswith(f"{p}=") for p in _RAY_KIT_ARGV_PREFIXES):
            removed.append(arg)
            continue
        kept.append(arg)
    if removed:
        sys.argv = kept
    return removed


class CloudpickleWrapper:
    """
    transform complex object like function between processes.
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternion to axis-angle format, inspired from libero utils, batch version.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    bs = quat.shape[0]
    quat_clipped = quat.clone()
    quat_clipped[:, 3] = torch.clamp(quat[:, 3], -1, 1)

    den = torch.sqrt(1.0 - quat_clipped[:, 3] * quat_clipped[:, 3])
    zero_pos = torch.isclose(den, torch.zeros(bs, device=quat.device))
    unscale_axisangle = quat_clipped[:, :3] * 2 * torch.acos(quat_clipped[:, 3:])

    return torch.where(
        zero_pos.unsqueeze(1),
        torch.zeros_like(unscale_axisangle, device=quat.device),
        unscale_axisangle / den.unsqueeze(1),
    )
