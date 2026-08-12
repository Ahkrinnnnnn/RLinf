#!/usr/bin/env bash
# Batch open-loop replays for the three RC09 subtasks (ep0 by default).
# Requires My Passport mounted at /media/lenovo/My Passport.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../../../" && pwd)"
cd "$ROOT"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$ROOT/.venv/isaaclab}"
OUT="${1:-$ROOT/logs/replay}"
EP="${2:-0}"
mkdir -p "$OUT"

PASSPORT="/media/lenovo/My Passport"
if [[ ! -d "$PASSPORT" ]]; then
  echo "ERROR: $PASSPORT not mounted. Re-plug the USB Passport (/dev/sdb1) first."
  exit 1
fi

run_one() {
  local sub="$1"
  local ds="$2"
  local tag="$3"
  echo "======== replay $sub ep$EP ========"
  python rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/replay_lerobot_episode.py \
    --subtask "$sub" \
    --dataset "$PASSPORT/$ds" \
    --episode "$EP" \
    --headless --enable_cameras \
    --video-out "$OUT/${tag}_ep${EP}.mp4" \
    --csv-out "$OUT/${tag}_ep${EP}.csv"
}

run_one insert_tube insert_the_blue_tube_all tube
run_one insert_rod insert_the_rod_all rod
run_one stack_base stack_the_base_all stack

echo "Done. Inspect $OUT/*_ep${EP}.csv grip events + videos."
