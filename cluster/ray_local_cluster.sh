#!/usr/bin/env bash

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# parse args from `args_parser.sh` getopts
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/args_parser.sh"

tmpdir=/tmp/symlink_$(uuidgen | cut -d "-" -f5)
echo "Create symlink: $outdir -> $tmpdir"

scratch_root=/scratch/$USER
training_scratch="${scratch_root}/training"
sandbox_dir="${scratch_root}/pgdb/sandbox"
mkdir -p "$training_scratch" "${scratch_root}/pgdb"

############################## SETUP PORTS
# for debugging
set -x

#bias to selection of higher range ports
function getfreeport()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 )  + 20000 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

port=$(getfreeport)
echo "Head node will use port: $port"
export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"
export dashboard_port

############################## FIND NODES/HOSTS

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

: "${SUPABASE_LOCAL_PORT:?SUPABASE_LOCAL_PORT must be set in the environment}"

wait_for_local_db_from_training_image() {
    local host=$1
    local port=$2
    local uri=$3
    local attempts=${4:-30}

    for ((attempt=1; attempt<=attempts; attempt++)); do
        if apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $outdir:$tmpdir \
            --bind $training_scratch:/scratch \
            $env bash -lc "pg_isready -h \"$host\" -p \"$port\" -d postgres >/dev/null 2>&1 && psql \"$uri\" --command='SELECT 1;'" >/dev/null 2>&1; then
            return 0
        fi
        echo "Waiting for local database from training image at ${host}:${port} (attempt ${attempt}/${attempts})"
        sleep 2
    done

    echo "Training image could not reach local database at ${host}:${port}" >&2
    apptainer exec --userns --nv \
        --bind $storage_server \
        --bind $workspace \
        --bind $bind \
        --bind $outdir:$tmpdir \
        --bind $training_scratch:/scratch \
        $env bash -lc "pg_isready -h \"$host\" -p \"$port\" -d postgres || true; psql \"$uri\" --command='SELECT 1;' || true" || true
    exit 1
}

############################## START HEAD NODE

echo "Copying local database to head $head_node"
rsync -avz --stats "$database_sandbox" "${scratch_root}/pgdb/sandbox.tar.zst"
tar --zstd -xf "${scratch_root}/pgdb/sandbox.tar.zst" -C "${scratch_root}/pgdb"

SUPABASE_LOCAL_HOST="$head_node_ip"
SUPABASE_LOCAL_URI="postgresql://postgres:postgres@${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}/postgres"
export SUPABASE_LOCAL_HOST
export SUPABASE_LOCAL_URI

echo "Starting local database from ray_local_cluster.sh"
apptainer instance stop mysql >/dev/null 2>&1 || true
apptainer instance start --no-mount proc --writable --env POSTGRES_PASSWORD=postgres "$sandbox_dir" mysql
apptainer exec instance://mysql postgres \
    -c "listen_addresses=0.0.0.0" \
    -c "port=${SUPABASE_LOCAL_PORT}" \
    -c 'config_file=/etc/postgresql/postgresql.conf' &
db_pid=$!
wait_for_local_db_from_training_image "$SUPABASE_LOCAL_HOST" "$SUPABASE_LOCAL_PORT" "$SUPABASE_LOCAL_URI"
echo "Training image can reach local database at ${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}"

apptainer exec --userns --nv \
    --bind $storage_server \
    --bind $workspace \
    --bind $bind \
    --bind /dev/shm:/dev/shm \
    --bind $training_scratch:/scratch \
    --bind $outdir:$tmpdir \
    $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
    -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $gpus -t $tmpdir -q $object_store_memory &
sleep 60

check_headnode="apptainer exec --nv \
    --bind $storage_server --bind $workspace \
    --bind $bind \
    --bind $outdir:$tmpdir \
    --bind $training_scratch:/scratch \
    $env ray status --address $head_node_ip:$port"
while ! $check_headnode; do
    echo "Waiting for head node..."
    sleep 3
done

rpids=$(pgrep -u $USER ray)
echo "Ray head node PID:"
echo $rpids

############################## CLEANUP

cleanup() {
    ec=$?
    echo "Running job cleanup (exit code: $ec)"
    head_pid=$(cat "$outdir/cleanup_head.pid" 2>/dev/null || true)
    if [[ -n "$head_pid" ]]; then
        kill -TERM "$head_pid" 2>/dev/null || true
        for _ in {1..120}; do
            kill -0 "$head_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$head_pid" 2>/dev/null || true
    fi
    apptainer instance stop mysql >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' SIGTERM SIGINT

############################## CHECK STATUS

echo apptainer exec --userns --nv \
    --bind $storage_server \
    --bind $workspace \
    --bind $bind \
    --bind $outdir:$tmpdir \
    --bind $training_scratch:/scratch \
    $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
    -a $cluster_address -r 1

############################## PRE-TRAINING DB HEALTH CHECK

echo "=== Active Apptainer instances ==="
apptainer instance list
echo "=== DB reachability check before training ==="
if apptainer exec --userns --nv \
    --bind $storage_server --bind $workspace --bind $bind \
    --bind $outdir:$tmpdir --bind $training_scratch:/scratch \
    $env bash -lc "pg_isready -h \"$SUPABASE_LOCAL_HOST\" -p \"$SUPABASE_LOCAL_PORT\" -d postgres"; then
    echo "DB is reachable at ${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}"
else
    echo "WARNING: DB is NOT reachable at ${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}" >&2
    apptainer instance list
fi

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv \
    --bind $storage_server \
    --bind $workspace \
    --bind $bind \
    --bind $outdir:$tmpdir \
    --bind $training_scratch:/scratch \
    $env $tasks