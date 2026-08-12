#!/usr/bin/env python3
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

"""Open-loop replay of a LeRobot CRP episode in Isaac-RC09-PegInsert-Visuomotor.

Example::

  export RC09_SUBTASK=insert_tube
  export ISAACLAB_PATH=/home/lenovo/zyy/RLinf/.venv/isaaclab
  cd /home/lenovo/zyy/RLinf
  python rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/replay_lerobot_episode.py \\
    --dataset "/media/lenovo/My Passport/insert_the_blue_tube_all" \\
    --episode 0 --headless --enable_cameras \\
    --video-out logs/replay/tube_ep0.mp4 \\
    --csv-out logs/replay/tube_ep0.csv

Actions are absolute joint angles (deg) + GOT0; converted via ``policy_action_to_sim``.
If the arm folds the wrong way, try ``--joint-signs 1,-1,-1,1,1,1``.

Notes on frames:
  * Joint replay ignores the lab **user frame** and pendant tool offset.
  * Active TCP is a fixed ``(0,0,0.2)`` along EE +Z (approx only for EE logs /
    wrist cam). Object XY alignment should prefer visual / side-slot layout.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# AppLauncher must start before most Isaac / project imports.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="LeRobot dataset root (default: Passport path for RC09_SUBTASK)",
)
parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
parser.add_argument("--max-steps", type=int, default=0, help="Cap steps (0 = full episode)")
parser.add_argument(
    "--joint-signs",
    type=str,
    default=None,
    help="Comma-separated CRP→sim signs for j1..j6 (also sets RC09_JOINT_SIGNS for home)",
)
parser.add_argument(
    "--subtask",
    type=str,
    default=None,
    help="insert_rod | insert_tube | stack_base (sets RC09_SUBTASK before env import)",
)
parser.add_argument("--video-out", type=str, default="", help="Optional mp4 path (table_cam)")
parser.add_argument(
    "--wrist-video-out",
    type=str,
    default="",
    help="Optional mp4 for wrist_cam (empty = derive from --video-out)",
)
parser.add_argument(
    "--csv-out",
    type=str,
    default="",
    help="Per-step EE/object/gripper CSV (empty = derive from --video-out)",
)
parser.add_argument("--fps", type=float, default=30.0, help="Dataset / video fps")
parser.add_argument("--task", type=str, default="Isaac-RC09-PegInsert-Visuomotor-v0")
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher = None  # filled after isaaclab import path is ready

_ARM_JOINT_NAMES = (
    "J1_Joint",
    "J2_Joint",
    "J3_joint",
    "J4_Joint",
    "J5_Joint",
    "J6_Joint",
)
_OBJECT_KEYS = ("base", "middle", "tube", "peg")


def _early_env_from_args(args: argparse.Namespace) -> None:
    if args.subtask:
        os.environ["RC09_SUBTASK"] = args.subtask
    if args.joint_signs:
        os.environ["RC09_JOINT_SIGNS"] = args.joint_signs


def _parse_signs(raw: str | None) -> tuple[float, float, float, float, float, float] | None:
    if not raw:
        env_raw = os.environ.get("RC09_JOINT_SIGNS")
        if not env_raw:
            return None
        raw = env_raw
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 6:
        raise SystemExit(f"--joint-signs / RC09_JOINT_SIGNS need 6 floats, got {raw!r}")
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])


def _load_episode_actions(dataset_root: Path, episode: int):
    import numpy as np
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {dataset_root / 'data'}")

    actions: list[np.ndarray] = []
    states0: np.ndarray | None = None
    for path in files:
        table = pq.read_table(
            path,
            columns=["action", "observation.state", "episode_index", "frame_index"],
        )
        df = table.to_pandas()
        mask = df["episode_index"].astype(int) == int(episode)
        if not mask.any():
            continue
        ep = df.loc[mask].sort_values("frame_index")
        for _, row in ep.iterrows():
            if int(row["frame_index"]) == 0:
                states0 = np.asarray(row["observation.state"], dtype=np.float32)
            actions.append(np.asarray(row["action"], dtype=np.float32))
        break
    if not actions:
        raise KeyError(f"Episode {episode} not found in {dataset_root}")
    if states0 is None:
        states0 = actions[0][:6].copy()
    return np.stack(actions, axis=0), states0


def _got0_is_open(got0: float, thresh: float = 500.0) -> bool:
    return float(got0) >= thresh


def _grip_events(actions_np) -> list[tuple[int, str, float]]:
    """Detect GOT0 open↔close transitions (dataset action[:,6])."""
    events: list[tuple[int, str, float]] = []
    prev = _got0_is_open(float(actions_np[0, 6]))
    events.append((0, "start_open" if prev else "start_closed", float(actions_np[0, 6])))
    for i in range(1, len(actions_np)):
        cur = _got0_is_open(float(actions_np[i, 6]))
        if cur == prev:
            continue
        kind = "open" if cur else "close"
        events.append((i, kind, float(actions_np[i, 6])))
        prev = cur
    return events


def _tensor_xyz(t) -> tuple[float, float, float]:
    import numpy as np

    a = t.detach().float().cpu().numpy().reshape(-1)
    return (float(a[0]), float(a[1]), float(a[2]))


def _read_ee_tcp(unwrapped) -> tuple[float, float, float]:
    ee = unwrapped.scene["ee_frame"]
    pos = ee.data.target_pos_w[0, 0] - unwrapped.scene.env_origins[0]
    return _tensor_xyz(pos)


def _read_object_xyz(unwrapped, name: str) -> tuple[float, float, float] | None:
    if name not in unwrapped.scene.keys():
        return None
    try:
        asset = unwrapped.scene[name]
        pos = asset.data.root_pos_w[0] - unwrapped.scene.env_origins[0]
        return _tensor_xyz(pos)
    except Exception:  # noqa: BLE001
        return None


def _img_from_obs(obs, key: str):
    import numpy as np

    policy = obs.get("policy", obs) if isinstance(obs, dict) else {}
    if not isinstance(policy, dict) or key not in policy:
        return None
    img = policy[key][0]
    arr = img.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr


def _save_video(path: Path, frames: list, fps: float) -> None:
    if not frames:
        print(f"[replay] no frames for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
        print(f"[replay] wrote {path} ({len(frames)} frames)")
    except Exception as exc:  # noqa: BLE001
        out_dir = path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(out_dir / f"{i:05d}.png")
        print(f"[replay] imageio failed ({exc}); wrote PNGs under {out_dir}")


def main() -> None:
    args, app_args = parser.parse_known_args()
    _early_env_from_args(args)

    # Ensure RLinf + Isaac Lab task package are importable.
    rlinf_root = Path(__file__).resolve().parents[6]
    sys.path.insert(0, str(rlinf_root))
    isaaclab = os.environ.get("ISAACLAB_PATH", str(rlinf_root / ".venv" / "isaaclab"))
    sys.path.insert(0, str(Path(isaaclab) / "source" / "isaaclab_tasks"))
    sys.path.insert(0, str(Path(isaaclab) / "source" / "isaaclab"))

    from isaaclab.app import AppLauncher as _AppLauncher

    _AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _early_env_from_args(args)
    joint_signs = _parse_signs(args.joint_signs)

    app_launcher = _AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.camera_calibration import (
        RC09_TCP_OFFSET_POS_M,
    )
    from rlinf.envs.isaaclab.isaaclab_tasks.rc09_peg_insert.subtasks import (
        default_dataset_root,
        resolve_active_subtask,
    )
    from rlinf.envs.isaaclab.rc09_real_robot_io import policy_action_to_sim

    subtask = resolve_active_subtask(args.subtask)
    dataset_root = Path(args.dataset or default_dataset_root(subtask)).expanduser()
    print(f"[replay] subtask={subtask.id} prompt={subtask.prompt!r}")
    print(f"[replay] dataset={dataset_root} episode={args.episode}")
    print(f"[replay] tcp_offset_m={RC09_TCP_OFFSET_POS_M} (approx EE log only)")

    if not dataset_root.is_dir():
        raise SystemExit(
            f"[replay] dataset missing: {dataset_root}\n"
            "  Re-plug My Passport (USB /dev/sdb1 → /media/lenovo/My Passport) and retry."
        )

    actions_np, state0 = _load_episode_actions(dataset_root, args.episode)
    if args.max_steps > 0:
        actions_np = actions_np[: args.max_steps]
    events = _grip_events(actions_np)
    print(f"[replay] frames={len(actions_np)} state0_deg={state0.round(2)}")
    print(f"[replay] gripper transitions ({len(events)}):")
    for step, kind, got0 in events:
        print(f"  step={step:4d}  {kind:12s}  GOT0={got0:.1f}")

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    if hasattr(env_cfg, "set_domain_randomization"):
        env_cfg.set_domain_randomization(False)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    # Snap arm to episode first-frame state (deg → sim rad, with optional signs).
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    q0 = policy_action_to_sim(
        torch.as_tensor(
            np.concatenate([state0, [1000.0]])[None, :],
            device=unwrapped.device,
            dtype=torch.float32,
        ),
        joint_signs=joint_signs,
    )[0, :6]
    joint_ids = [robot.joint_names.index(n) for n in _ARM_JOINT_NAMES]
    full_q = robot.data.default_joint_pos.clone()
    full_q[:, joint_ids] = q0
    robot.write_joint_state_to_sim(full_q, torch.zeros_like(full_q))
    robot.set_joint_position_target(full_q)
    unwrapped.sim.forward()

    video_path = Path(args.video_out).expanduser() if args.video_out else None
    if args.wrist_video_out:
        wrist_path = Path(args.wrist_video_out).expanduser()
    elif video_path is not None:
        wrist_path = video_path.with_name(video_path.stem + "_wrist" + video_path.suffix)
    else:
        wrist_path = None

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser()
    elif video_path is not None:
        csv_path = video_path.with_suffix(".csv")
    else:
        csv_path = Path("logs/replay") / f"{subtask.id}_ep{args.episode}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table_frames: list = []
    wrist_frames: list = []
    event_by_step = {s: k for s, k, _ in events}
    rows: list[dict] = []

    try:
        for i, act in enumerate(actions_np):
            if not simulation_app.is_running():
                break
            action_t = policy_action_to_sim(
                torch.as_tensor(act[None, :], device=unwrapped.device, dtype=torch.float32),
                joint_signs=joint_signs,
            )
            obs, _, _, _, _ = env.step(action_t)

            ee = _read_ee_tcp(unwrapped)
            got0 = float(act[6]) if len(act) > 6 else float("nan")
            open_flag = int(_got0_is_open(got0)) if got0 == got0 else -1
            row: dict = {
                "step": i,
                "got0": got0,
                "gripper_open": open_flag,
                "event": event_by_step.get(i, ""),
                "ee_x": ee[0],
                "ee_y": ee[1],
                "ee_z": ee[2],
            }
            for name in _OBJECT_KEYS:
                xyz = _read_object_xyz(unwrapped, name)
                if xyz is None:
                    row[f"{name}_x"] = ""
                    row[f"{name}_y"] = ""
                    row[f"{name}_z"] = ""
                else:
                    row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = xyz
            rows.append(row)

            if i in event_by_step:
                kind = event_by_step[i]
                print(
                    f"[replay][{kind}] step={i} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                    f"GOT0={got0:.0f}"
                )
                for name in _OBJECT_KEYS:
                    xyz = _read_object_xyz(unwrapped, name)
                    if xyz is not None:
                        print(f"         {name}=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})")

            if video_path is not None:
                img = _img_from_obs(obs, "table_cam")
                if img is not None:
                    table_frames.append(img)
            if wrist_path is not None:
                img = _img_from_obs(obs, "wrist_cam")
                if img is not None:
                    wrist_frames.append(img)

            if (i + 1) % 50 == 0:
                print(f"[replay] step {i + 1}/{len(actions_np)} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
    finally:
        env.close()
        simulation_app.close()

    fieldnames = [
        "step",
        "got0",
        "gripper_open",
        "event",
        "ee_x",
        "ee_y",
        "ee_z",
    ]
    for name in _OBJECT_KEYS:
        fieldnames.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[replay] wrote {csv_path} ({len(rows)} rows)")

    # Compact summary of grasp/place EE (first close + subsequent open).
    closes = [r for r in rows if r["event"] == "close"]
    opens = [r for r in rows if r["event"] == "open"]
    if closes:
        c = closes[0]
        print(
            f"[replay] FIRST CLOSE tip≈({c['ee_x']:.3f},{c['ee_y']:.3f},{c['ee_z']:.3f}) "
            f"→ candidate object pick XY"
        )
    if opens:
        o = opens[-1] if len(opens) >= 1 else opens[0]
        print(
            f"[replay] LAST OPEN tip≈({o['ee_x']:.3f},{o['ee_y']:.3f},{o['ee_z']:.3f}) "
            f"→ candidate place / insert XY"
        )

    if video_path is not None:
        _save_video(video_path, table_frames, args.fps)
    if wrist_path is not None:
        _save_video(wrist_path, wrist_frames, args.fps)


if __name__ == "__main__":
    main()
