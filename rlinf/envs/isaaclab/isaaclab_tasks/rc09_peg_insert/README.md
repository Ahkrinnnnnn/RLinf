# RC09 peg-insert Isaac Lab task

Install (symlink into Isaac Lab fork; run once per machine / ISAACLAB_PATH):

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export RC09_URDF_DIR=/path/to/RC09-05_urdf.SLDASM
bash scripts/install_rc09_task.sh
```

Gym ID: `Isaac-RC09-PegInsert-Visuomotor-v0`  
Prompt: `Insert the pink rod into the blue tube`

Task source of truth is this directory in RLinf. The install script symlinks it into
Isaac Lab; edits here take effect without re-running install.

RL training configs: `examples/embodiment/config/rc09_peg_insert/README.md`
