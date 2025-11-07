#!/usr/bin/env bash
set -euo pipefail

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

if [ -z "${RANK:-}" ] || [ -z "${WORLD_SIZE:-}" ]; then
    echo "RANK and WORLD_SIZE not set in the environment."
    echo "Assuming single-node run with RANK=0 and WORLD_SIZE=${gpus}"
    RANK=0
    WORLD_SIZE=${gpus}
fi

if [ -n "${JOB_NAME:-}" ] && [ -n "${PROJECT_NAME:-}" ]; then
    echo "Running in Run:AI job ${JOB_NAME} in project ${PROJECT_NAME}"
else
    echo "JOB_NAME or PROJECT_NAME not set; stopping job."
    exit 1
fi

############################## SETUP PORTS

# for debugging
set -x

#bias to selection of higher range ports
getfreeport() {
  while :; do
    port=$(( (RANDOM % 40000) + 20000 ))
    if python3 - "$port" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)  # in use
finally:
    try:
        s.close()
    except Exception:
        pass
PY
    then
      echo "$port"
      return 0
    fi
  done
}

############################## HELPERS

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
    uv run ray stop --force >/dev/null 2>&1 || true
    ) >/dev/null 2>&1 || true

    echo "[RANK ${RANK}]: Exiting job."
    exit 0

    # only rank 0 tears the job down (assumes AI CLI is available)
    # if [ "${RANK}" -eq 0 ]; then
    #     echo "Rank 0 deleting job ${JOB_NAME} in project ${PROJECT_NAME}"
    #     ai job delete --name "$JOB_NAME" --project "${PROJECT_NAME}"
    # fi
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
    bash -lc "bash /work/cell_observatory_platform/cluster/ray_start_cluster_runai.sh \
        -i \"$head_node_ip\" -p \"$port\" -d \"$dashboard_port\" \
        -c \"${head_cpus}\" -g \"${head_gpus}\" -t \"$outdir\" -q \"${object_store_memory}\"" &

    sleep 10

    bash -lc "/work/cell_observatory_platform/cluster/ray_check_status_runai.sh -a \"$cluster_address\" -r 1"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[rank=$RANK] Head failed to start; rc=$rc"
        exit $rc
    fi
    echo "[rank=$RANK] Head healthy at $cluster_address"
    echo "$cluster_address" > "$outdir/cluster_address"
else
    deadline=$((SECONDS + 300))
    while [ ! -s "$outdir/cluster_address" ]; do
        (( SECONDS >= deadline )) && { echo "[rank=$RANK] Timeout waiting for cluster_address"; exit 1; }
        sleep 2
    done
    cluster_address="$(cat "$outdir/cluster_address")"

    if [[ -z "${cluster_address:-}" ]]; then
        echo "[rank=$RANK] cluster_address is empty; refusing to start worker."
        exit 1
    fi

    worker_index=$((RANK - 1))
    mkdir -p "$outdir/ray_worker_${worker_index}"

    echo "[rank=$RANK] Starting Ray worker idx=$worker_index -> head at $cluster_address"
    bash -lc "exec /work/cell_observatory_platform/cluster/ray_start_worker_runai.sh \
        -a \"$cluster_address\" \
        -c \"${cpus}\" \
        -g \"${gpus}\" \
        -t \"$outdir\" \
        -q \"${object_store_memory}\" \
        -w \"${worker_index}\""
fi

############################## CLUSTER HEALTH

bash -lc "/work/cell_observatory_platform/cluster/ray_check_status_runai.sh -a \"$cluster_address\" -r \"$WORLD_SIZE\""
rc=$?
if [ $rc -ne 0 ]; then
    echo "Cluster failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

############################## RUN WORKLOAD

echo "Running user tasks: ${tasks:-}"
if [ -n "${tasks:-}" ]; then
    bash -lc "$tasks"
else
    echo "No tasks specified!"
fi

############################## CLEANUP

echo "User tasks completed, starting cleanup"
do_cleanup
exit 0