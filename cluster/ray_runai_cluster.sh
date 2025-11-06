#!/usr/bin/env bash
set -euo pipefail

# remove when i figure out what env vars are called
echo "=== All environment variables ==="
printenv
echo "==============================="

# NCCL settings optimized for Ethernet without InfiniBand
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# parse args from `args_parser.sh` getopts
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/args_parser.sh"

mkdir -p "$outdir"

# only rank 0 writes the submitted config
if [ "${RANK:-0}" -eq 0 ] && [ -n "${CFG_B64:-}" ]; then
  python3 - <<'PY'
import os, base64, zlib, pathlib
outdir = os.environ.get("CFG_SAVEDIR")
exp_name = os.environ.get("EXP_NAME")
p = pathlib.Path(outdir).expanduser().resolve()
p.mkdir(parents=True, exist_ok=True)
b64 = os.environ["CFG_B64"]
data = zlib.decompress(base64.b64decode(b64.encode("ascii")))
dest = p / exp_name
dest.write_bytes(data)
print(f"[cfg] wrote {dest}")
PY
fi

sleep 10

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

wait_for_head() {
    local addr="$1"; local timeout="${2:-180}"; local interval="${3:-2}"
    local host="${addr%:*}"; local port="${addr##*:}"
    local deadline=$((SECONDS + timeout))
    echo "[wait] Waiting for head at ${host}:${port} (timeout=${timeout}s, interval=${interval}s)"
    while (( SECONDS < deadline )); do
        if exec 3<>"/dev/tcp/${host}/${port}" 2>/dev/null; then
            exec 3>&- 3<&-
            echo "[wait] Head is reachable at ${host}:${port}"
            return 0
        fi
        sleep "${interval}"
    done
    echo "[wait] Timed out waiting for head at ${host}:${port} after ${timeout}s"
    return 1
}

############################## CLEANUP

do_cleanup() {
    echo "Running cleanup (rank=$RANK)"

    if [ "$RANK" -eq 0 ]; then
        PF="$outdir/cleanup_head.pid"
    else
        worker_index=$((RANK - 1))
        PF="$outdir/cleanup_${worker_index}.pid"
    fi

    (
    GRACE_SECONDS=60
    if [ -f "$PF" ]; then
        pid=$(cat "$PF")
        kill -TERM "$pid" 2>/dev/null || true
        for ((i=0;i<GRACE_SECONDS;i++)); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
    fi
    python3 /work/cell_observatory_platform/utils/cleanup.py || true
    ray stop --force >/dev/null 2>&1 || true
    ) >/dev/null 2>&1 || true

    # only rank 0 tears the job down
    if [ "${RANK}" -eq 0 ] && [ -n "${JOB_NAME}" ] && [ -n "${PROJECT_NAME}" ]; then
        echo "Rank 0 deleting job ${JOB_NAME} in project ${PROJECT_NAME}"
        ai job delete --name "$JOB_NAME" --project "${PROJECT_NAME}"
    fi
}

trap 'do_cleanup; exit 130' INT    # SIGINT
trap 'do_cleanup; exit 143' TERM   # SIGTERM

############################## HEAD / WORKERS

if [ "$RANK" -eq 0 ]; then
    echo "[rank=$RANK] Electing self as head."

    port=$(getfreeport)
    dashboard_port=$(getfreeport)
    echo "Head will use: ray=$port, dashboard=$dashboard_port"

    head_node_ip="$(hostname -I | awk '{print $1}')"
    export RAY_GRAFANA_HOST="${head_node_ip}:3000"
    export RAY_PROMETHEUS_HOST="${head_node_ip}:9090"

    cluster_address="${head_node_ip}:${port}"
    export head_node_ip cluster_address

    echo "[rank=$RANK] Starting Ray head at $cluster_address (dashboard $dashboard_port)"
    bash -lc "/work/cell_observatory_platform/cluster/ray_start_cluster_runai.sh \
        -i \"$head_node_ip\" -p \"$port\" -d \"$dashboard_port\" \
        -c \"${head_cpus}\" -g \"${head_gpus}\" -t \"$outdir\" -q \"${object_store_memory}\"" &

    sleep 10

    bash -lc "/work/cell_observatory_platform/cluster/ray_check_status.sh -a \"$cluster_address\" -r 1"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[rank=$RANK] Head failed to start; rc=$rc"
        exit $rc
    fi
    echo "[rank=$RANK] Head healthy at $cluster_address"
else
    if [ -n "${CLUSTER_ADDRESS}" ]; then
        cluster_address="$CLUSTER_ADDRESS"
    elif [ -n "${MASTER_ADDR}" ] && [ -n "${MASTER_PORT}" ]; then
        cluster_address="${MASTER_ADDR}:${MASTER_PORT}"
    else
        echo "[rank=$RANK] ERROR: CLUSTER_ADDRESS or MASTER_ADDR/MASTER_PORT must be set."
        exit 1
    fi

    export cluster_address

    wait_for_head "$cluster_address" 180 2 || { echo "[rank=$RANK] Head not ready; exiting"; exit 1; }

    worker_index=$((RANK - 1))
    mkdir -p "$outdir/ray_worker_${worker_index}"

    echo "[rank=$RANK] Starting Ray worker idx=$worker_index -> head at $cluster_address"
    bash -lc "exec /work/cell_observatory_platform/cluster/ray_start_worker.sh \
        -a \"$cluster_address\" \
        -c \"${cpus}\" \
        -g \"${gpus}\" \
        -t \"$outdir\" \
        -q \"${object_store_memory}\" \
        -w \"${worker_index}\""
fi

############################## CLUSTER HEALTH

bash -lc "/work/cell_observatory_platform/cluster/ray_check_status.sh -a \"$cluster_address\" -r \"$WORLD_SIZE\""
rc=$?
if [ $rc -ne 0 ]; then
    echo "Cluster failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

############################## RUN WORKLOAD

echo "Running user tasks: $tasks"
if [ -n "${tasks:-}" ]; then
    bash -lc "$tasks"
else
    echo "No tasks specified!"
fi

############################## CLEANUP

echo "User tasks completed, starting cleanup"
do_cleanup
exit 0