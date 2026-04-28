#!/usr/bin/env bash
set -euo pipefail

# NOTE: assumes you are already in apptainer image for example:
# apptainer shell --nv --bind /tmp:/scratch --bind /dev/shm:/dev/shm --bind /clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed /clusterfs/nvme/martinalvarez/apptainer_images/feature_local_db_torch_26_01.sif

# RUN: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/db/local_start_sandbox.sh

DB_SANDBOX="/clusterfs/nvme/hph/git_managed/databases/2026_04_21_sandbox.tar.zst"

pip install -r /clusterfs/nvme/hph/git_managed/ml-data-pipeline/requirements.txt
repo_root="/clusterfs/nvme/hph/git_managed/cell_observatory_platform"
cd "$repo_root"

if [[ -f .env ]]; then
  source .env
fi

: "${DB_SANDBOX:?DB_SANDBOX must be set}"
: "${SUPABASE_LOCAL_PORT:?SUPABASE_LOCAL_PORT must be set}"
: "${REPO_DIR:?REPO_DIR must be set}"
: "${DATA_DIR:?DATA_DIR must be set}"
: "${STORAGE_SERVER_DIR:?STORAGE_SERVER_DIR must be set}"
: "${DATABASE_DIR:?DATABASE_DIR must be set}"
: "${NODE_LOCAL_STORE_ROOT:?NODE_LOCAL_STORE_ROOT must be set}"

scratch_root="${SCRATCH_ROOT:-$(dirname "$NODE_LOCAL_STORE_ROOT")}"
sandbox_tar="${scratch_root}/sandbox.tar.zst"
sandbox_dir="${SANDBOX_DIR:-${scratch_root}/sandbox}"
instance_name="${SANDBOX_INSTANCE_NAME:-sandbox_pg}"
wait_seconds="${SANDBOX_WAIT_SECONDS:-60}"

resolve_node_ip() {
  local raw_ip
  raw_ip=$(hostname --ip-address 2>/dev/null || true)
  raw_ip=${raw_ip%% *}
  if [[ -z "$raw_ip" ]]; then
    echo "Failed to resolve node IP for local sandbox"
    exit 1
  fi
  echo "$raw_ip"
}

SUPABASE_LOCAL_HOST="${SUPABASE_LOCAL_HOST:-$(resolve_node_ip)}"
SUPABASE_LOCAL_URI="postgresql://postgres:postgres@${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}/postgres"
export SUPABASE_LOCAL_HOST
export SUPABASE_LOCAL_URI

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found"; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "psql not found"; exit 1; }
command -v rsync >/dev/null 2>&1 || { echo "rsync not found"; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "tar not found"; exit 1; }

mkdir -p "$scratch_root" "$NODE_LOCAL_STORE_ROOT"

echo "Stopping old instance if present: $instance_name"
apptainer instance stop "$instance_name" >/dev/null 2>&1 || true

echo "Removing old extracted sandbox: $sandbox_dir"
# rm -rf "$sandbox_dir"

echo "Copying sandbox tarball to $sandbox_tar"
rsync -av --no-group --progress "$DB_SANDBOX" "$sandbox_tar"

echo "Extracting sandbox under $scratch_root"
tar --zstd -xf "$sandbox_tar" -C "$scratch_root"

if [[ ! -d "$sandbox_dir" ]]; then
  echo "Expected extracted sandbox at $sandbox_dir but it was not found"
  exit 1
fi

# FIXME: all these are not needed
bind_dests=(
  "/global"
  "/clusterfs"
  "/workspace/cell_observatory_platform"
  "$DATA_DIR"
  "$STORAGE_SERVER_DIR"
  "$DATABASE_DIR"
  "$NODE_LOCAL_STORE_ROOT"
  "/scratch"
  "/dev/shm"
)

for dest in "${bind_dests[@]}"; do
  mkdir -p "${sandbox_dir}${dest}"
done

echo "Starting Postgres Apptainer instance: $instance_name"
env -u APPTAINER_BIND \
    -u APPTAINER_BINDPATH \
    -u SINGULARITY_BIND \
    -u SINGULARITY_BINDPATH \
    apptainer instance start \
      --no-mount proc \
      --writable \
      --env POSTGRES_PASSWORD=postgres \
      "$sandbox_dir" \
      "$instance_name"

echo "Launching Postgres inside instance: $instance_name"
apptainer exec instance://"$instance_name" postgres \
    -c "listen_addresses=0.0.0.0" \
    -c "port=${SUPABASE_LOCAL_PORT}" \
    -c 'config_file=/etc/postgresql/postgresql.conf' &

echo "Waiting for Postgres on $SUPABASE_LOCAL_URI"
ready=0
for ((i=0; i<wait_seconds; i+=2)); do
  if psql "$SUPABASE_LOCAL_URI" --command="SELECT 1;" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  echo "Postgres did not become ready in ${wait_seconds}s"
  echo "Instance status:"
  apptainer instance list || true
  exit 1
fi

echo "Database is up"
psql "$SUPABASE_LOCAL_URI" --command="SELECT 1;"