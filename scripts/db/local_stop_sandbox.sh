#!/usr/bin/env bash
set -euo pipefail

repo_root="/workspace/cell_observatory_platform"
cd "$repo_root"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

: "${NODE_LOCAL_STORE_ROOT:?NODE_LOCAL_STORE_ROOT must be set}"

scratch_root="${SCRATCH_ROOT:-$(dirname "$NODE_LOCAL_STORE_ROOT")}"
sandbox_tar="${scratch_root}/sandbox.tar.zst"
sandbox_dir="${SANDBOX_DIR:-${scratch_root}/sandbox}"
instance_name="${SANDBOX_INSTANCE_NAME:-sandbox_pg}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found"; exit 1; }

echo "Stopping instance if present: $instance_name"
apptainer instance stop "$instance_name" >/dev/null 2>&1 || true

echo "Removing extracted sandbox: $sandbox_dir"
rm -rf "$sandbox_dir"

echo "Removing copied tarball: $sandbox_tar"
rm -f "$sandbox_tar"