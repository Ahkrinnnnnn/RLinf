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

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ..isaaclab_env import IsaaclabBaseEnv
from ..rc09_real_robot_io import policy_action_to_sim, sim_joint_pos_to_policy


def _tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """Convert env camera tensor [H,W,C] or [C,H,W] to uint8 HWC numpy."""
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _ensure_gui_display() -> str:
    """Resolve DISPLAY for Isaac GUI; fail fast if no X server socket exists."""
    display = os.environ.get("DISPLAY", "").strip()
    x_dir = Path("/tmp/.X11-unix")
    candidates: list[str] = []
    if display:
        candidates.append(display)
    # Local GDM sessions are often :1, not :0.
    if x_dir.is_dir():
        for sock in sorted(x_dir.glob("X[0-9]*")):
            candidates.append(f":{sock.name[1:]}")
    for cand in candidates:
        num = cand.split(":")[-1].split(".")[0]
        if (x_dir / f"X{num}").exists():
            os.environ["DISPLAY"] = cand if cand.startswith(":") else f":{num}"
            xauth = Path.home() / ".Xauthority"
            if xauth.is_file() and not os.environ.get("XAUTHORITY"):
                os.environ["XAUTHORITY"] = str(xauth)
            # AppLauncher falls back to HEADLESS env when headless=False.
            os.environ["HEADLESS"] = "0"
            return os.environ["DISPLAY"]
    raise RuntimeError(
        "headless=false but no X socket under /tmp/.X11-unix. "
        "On this machine the desktop session is typically DISPLAY=:1; set it "
        "explicitly before launching eval, or use headless=true."
    )


class IsaaclabRC09PegInsertEnv(IsaaclabBaseEnv):
    """RLinf wrapper for Isaac-RC09-PegInsert-Visuomotor-v0."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._startup_views_saved = False

    def _make_env_function(self):
        def make_env_isaaclab():
            headless = bool(self.cfg.init_params.get("headless", True))
            enable_cameras = bool(self.cfg.init_params.get("enable_cameras", True))
            # Headless Kit often breaks if DISPLAY is set to a bad GLX session.
            if headless:
                os.environ.pop("DISPLAY", None)
                os.environ["HEADLESS"] = "1"
            else:
                display = _ensure_gui_display()
                print(
                    f"[IsaaclabRC09PegInsertEnv] GUI mode: DISPLAY={display}, "
                    f"XAUTHORITY={os.environ.get('XAUTHORITY', '')}"
                )

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(
                headless=headless, enable_cameras=enable_cameras
            ).app
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
            isaac_env_cfg.scene.wrist_cam.height = (
                self.cfg.init_params.wrist_cam.height
            )
            isaac_env_cfg.scene.wrist_cam.width = (
                self.cfg.init_params.wrist_cam.width
            )
            isaac_env_cfg.scene.table_cam.height = (
                self.cfg.init_params.table_cam.height
            )
            isaac_env_cfg.scene.table_cam.width = (
                self.cfg.init_params.table_cam.width
            )
            if hasattr(isaac_env_cfg, "set_domain_randomization"):
                isaac_env_cfg.set_domain_randomization(
                    self.cfg.init_params.get("enable_domain_randomization", True)
                )
            elif hasattr(isaac_env_cfg, "enable_domain_randomization"):
                # Fallback if an older cfg lacks the helper; still flip the flag.
                isaac_env_cfg.enable_domain_randomization = (
                    self.cfg.init_params.get("enable_domain_randomization", True)
                )

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _startup_camera_dir(self) -> Path | None:
        if not self.cfg.init_params.get("save_startup_camera_views", True):
            return None
        if self.worker_info is not None and getattr(self.worker_info, "rank", 0) != 0:
            return None
        custom_dir = self.cfg.init_params.get("startup_camera_dir")
        if custom_dir:
            return Path(str(custom_dir)).expanduser()
        video_base = getattr(self.video_cfg, "video_base_dir", None)
        if video_base:
            return Path(str(video_base)).parent / "camera_startup"
        return Path("camera_startup")

    def _save_startup_camera_views(self, obs: dict) -> None:
        if self._startup_views_saved:
            return
        save_dir = self._startup_camera_dir()
        if save_dir is None:
            return

        try:
            from PIL import Image
        except ImportError:
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        top = _tensor_to_uint8_hwc(obs["policy"]["table_cam"][0])
        wrist = _tensor_to_uint8_hwc(obs["policy"]["wrist_cam"][0])
        Image.fromarray(top).save(save_dir / "top_cam_env0.png")
        Image.fromarray(wrist).save(save_dir / "wrist_cam_env0.png")
        self._startup_views_saved = True

    def step(self, actions=None, auto_reset=True):
        if actions is not None:
            actions = policy_action_to_sim(actions)
        return super().step(actions, auto_reset=auto_reset)

    def chunk_step(self, chunk_actions):
        return super().chunk_step(policy_action_to_sim(chunk_actions))

    def _wrap_obs(self, obs):
        self._save_startup_camera_views(obs)

        instruction = [self.task_description] * self.num_envs
        wrist_image = obs["policy"]["wrist_cam"]
        table_image = obs["policy"]["table_cam"]
        # Isaac internal: rad → LeRobot SFT / OpenPI: degrees (j1.pos … j6.pos)
        states = sim_joint_pos_to_policy(obs["policy"]["arm_joint_pos"])

        return {
            "main_images": table_image,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist_image,
        }
