#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

IMAGE="/groups/betzig/betziglab/hph/containers/cell_observatory_platform_feature-local_db_caches_torch_26_01.sif"

REPO_DIR="${REPO_DIR:?REPO_DIR must be set}"
DATA_DIR="${DATA_DIR:?DATA_DIR must be set}"
STORAGE_SERVER_DIR="${STORAGE_SERVER_DIR:?STORAGE_SERVER_DIR must be set}"
DATABASE_DIR="${DATABASE_DIR:?DATABASE_DIR must be set}"
NODE_LOCAL_STORE_ROOT="${NODE_LOCAL_STORE_ROOT:?NODE_LOCAL_STORE_ROOT must be set}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$(dirname "$NODE_LOCAL_STORE_ROOT")}"
mkdir -p "$SCRATCH_ROOT" "$NODE_LOCAL_STORE_ROOT"

apptainer shell --userns --nv \
  --bind "${REPO_DIR}:/workspace/cell_observatory_platform" \
  --bind "${DATA_DIR}:${DATA_DIR}" \
  --bind "${STORAGE_SERVER_DIR}:${STORAGE_SERVER_DIR}" \
  --bind "${DATABASE_DIR}:${DATABASE_DIR}" \
  --bind "${NODE_LOCAL_STORE_ROOT}:${NODE_LOCAL_STORE_ROOT}" \
  --bind "${SCRATCH_ROOT}:/scratch" \
  --bind /dev/shm:/dev/shm \
  "$IMAGE"