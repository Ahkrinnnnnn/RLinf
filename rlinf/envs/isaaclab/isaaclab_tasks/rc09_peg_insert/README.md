# RC09 tabletop Isaac Lab task (3 sequential subtasks, one CAD scene)

Install (symlink into Isaac Lab fork; run once per machine / ISAACLAB_PATH):

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export RC09_URDF_DIR=/path/to/RC09-05_urdf.SLDASM
bash scripts/install_rc09_task.sh
```

Gym ID: `Isaac-RC09-PegInsert-Visuomotor-v0`

## Assembly pipeline (start scene ≠ shared)

Each skill **starts from the state after prior skills**:

| Step | `RC09_SUBTASK` | Prompt | Start scene |
|------|----------------|--------|-------------|
| 1 | `stack_base` | Stack purple on black | purple **beside** empty black; tube/peg on table |
| 2 | `insert_tube` | Insert blue tube into purple | purple **already in** black; tube on table |
| 3 | `insert_rod` | Insert pink rod into blue tube | purple in black + tube **already seated** in purple; peg on table |

```bash
export RC09_SUBTASK=insert_tube   # or stack_base / insert_rod
```

Optional CRP→URDF joint signs (if replay folds the arm):

```bash
export RC09_JOINT_SIGNS=1,-1,-1,1,1,1
```

## Open-loop dataset replay

```bash
export ISAACLAB_PATH=$PWD/.venv/isaaclab
export RC09_SUBTASK=insert_tube
python rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/replay_lerobot_episode.py \
  --episode 0 --headless --enable_cameras \
  --video-out logs/replay_tube_ep0.mp4
```

Task source of truth is this directory in RLinf. The install script symlinks it into
Isaac Lab; edits here take effect without re-running install.

RL training configs: `examples/embodiment/config/rc09_peg_insert/README.md`
