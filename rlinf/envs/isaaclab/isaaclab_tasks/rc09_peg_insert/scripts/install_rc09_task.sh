#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
#
# Install RC09 peg-insert task into the Isaac Lab fork via symlink.
#
# The task source of truth lives in RLinf; Isaac Sim loads tasks from
# ISAACLAB_PATH. A symlink keeps edits in RLinf immediately visible without
# re-running this script. Re-run only on first setup or when ISAACLAB_PATH changes.
#
# Usage:
#   export ISAACLAB_PATH=/path/to/IsaacLab
#   export RC09_URDF_DIR=/path/to/RC09-05_urdf.SLDASM   # optional
#   bash rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SRC="$(realpath "${SCRIPT_DIR}/..")"
# scripts/ -> peg_insert -> isaaclab_tasks -> isaaclab -> envs -> rlinf -> RLinf
RLINF_ROOT="$(realpath "${SCRIPT_DIR}/../../../../../../")"

if [[ -z "${ISAACLAB_PATH:-}" ]]; then
  echo "ERROR: Set ISAACLAB_PATH to your Isaac Lab checkout." >&2
  exit 1
fi

TASK_DST="${ISAACLAB_PATH}/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/rc09_peg_insert"
URDF_SRC="${RC09_URDF_DIR:-${RLINF_ROOT}/../RC09-05_urdf.SLDASM}"
URDF_FILE="${URDF_SRC}/urdf/RC09-05_urdf.SLDASM.urdf"
PATCHED_URDF="${URDF_SRC}/urdf/RC09-05_urdf.SLDASM.isaac.urdf"

echo "Installing RC09 peg-insert task (symlink)..."
echo "  Source: ${TASK_SRC}"
echo "  Link:   ${TASK_DST}"

mkdir -p "$(dirname "${TASK_DST}")"
if [[ -e "${TASK_DST}" ]] || [[ -L "${TASK_DST}" ]]; then
  rm -rf "${TASK_DST}"
fi
ln -s "${TASK_SRC}" "${TASK_DST}"
echo "  Symlink created."

# Patch URDF package:// mesh paths for Isaac Sim (written next to source URDF).
if [[ -f "${URDF_FILE}" ]]; then
  MESH_DIR="$(realpath "${URDF_SRC}/meshes")"
  sed "s|package://RC09-05_urdf.SLDASM/meshes/|${MESH_DIR}/|g" "${URDF_FILE}" > "${PATCHED_URDF}"
  echo "  Patched URDF: ${PATCHED_URDF}"
else
  echo "WARNING: URDF not found at ${URDF_FILE}" >&2
  echo "         Set RC09_URDF_DIR or place RC09-05_urdf.SLDASM next to RLinf." >&2
fi

TASKS_INIT="${ISAACLAB_PATH}/source/isaaclab_tasks/isaaclab_tasks/__init__.py"
IMPORT_LINE="import isaaclab_tasks.manager_based.manipulation.rc09_peg_insert  # noqa: F401  # RC09 peg-insert"
if [[ -f "${TASKS_INIT}" ]]; then
  if ! grep -q "rc09_peg_insert" "${TASKS_INIT}"; then
    echo "${IMPORT_LINE}" >> "${TASKS_INIT}"
    echo "  Registered import in ${TASKS_INIT}"
  else
    echo "  Import already registered in ${TASKS_INIT}"
  fi
else
  echo "WARNING: ${TASKS_INIT} not found; add import manually." >&2
fi

echo "Done. Gym ID: Isaac-RC09-PegInsert-Visuomotor-v0"
echo "Note: Edit task code under RLinf; changes apply without re-running this script."
