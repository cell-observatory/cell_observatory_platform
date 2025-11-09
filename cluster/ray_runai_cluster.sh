#!/usr/bin/env bash

# NCCL settings optimized for Ethernet without InfiniBand
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
# export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# parse args from `args_parser.sh` getopts
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/args_parser.sh"

mkdir -p "$outdir"
# mkdir -p "$TMPDIR"

if [ -z "${RANK:-}" ]; then
    echo "RANK not set in the environment."
    echo "Assuming single-node run with RANK=0."
    RANK=0
fi

if [ -n "${JOB_NAME:-}" ]; then
    echo "Running in Run:AI job ${JOB_NAME}."
else
    echo "JOB_NAME not set; stopping job."
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

    echo "[RANK ${RANK}]: Exiting job." 
}

trap 'do_cleanup' EXIT   # normal exit
trap 'do_cleanup' INT    # SIGINT
trap 'do_cleanup' TERM   # SIGTERM

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
    export head_node_ip cluster_address port

    echo "[rank=$RANK] Starting Ray head at $cluster_address (dashboard $dashboard_port)"
    bash -lc "bash /work/cell_observatory_platform/cluster/ray_start_cluster_runai.sh \
        -i \"$head_node_ip\" -p \"$port\" -d \"$dashboard_port\" \
        -c \"${head_cpus}\" -g \"${head_gpus}\" -t \"$TMPDIR\" -o \"$outdir\"  -q \"${object_store_memory}\"" &

    sleep 10

    bash -lc "bash /work/cell_observatory_platform/cluster/ray_check_status_runai.sh -a \"$cluster_address\" -r 1"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[rank=$RANK] Head failed to start; rc=$rc"
        exit $rc
    fi
    echo "[rank=$RANK] Head healthy at $cluster_address"
    echo "$cluster_address" > "$outdir/cluster_address_${JOB_NAME}"
else
    deadline=$((SECONDS + 300))
    while [ ! -s "$outdir/cluster_address_${JOB_NAME}" ]; do
        (( SECONDS >= deadline )) && { echo "[rank=$RANK] Timeout waiting for cluster_address"; exit 1; }
        sleep 2
    done
    cluster_address="$(cat "$outdir/cluster_address_${JOB_NAME}")"

    if [[ -z "${cluster_address:-}" ]]; then
        echo "[rank=$RANK] cluster_address is empty; refusing to start worker."
        exit 1
    fi

    sleep 20

    worker_index=$((RANK - 1))
    mkdir -p "$outdir/ray_worker_${worker_index}"

    echo "[rank=$RANK] Starting Ray worker idx=$worker_index -> head at $cluster_address"
    bash -lc "bash /work/cell_observatory_platform/cluster/ray_start_worker_runai.sh \
        -a \"$cluster_address\" \
        -c \"${cpus}\" \
        -g \"${gpus}\" \
        -o \"$outdir\" \
        -t \"$TMPDIR\" \
        -q \"${object_store_memory}\" \
        -w \"${worker_index}\""
fi

############################## CLUSTER HEALTH

bash -lc "bash /work/cell_observatory_platform/cluster/ray_check_status_runai.sh -a \"$cluster_address\" -r \"$NUM_NODES\""
rc=$?
if [ $rc -ne 0 ]; then
    echo "Cluster failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

############################## RUN WORKLOAD

if [ "${RANK}" -eq 0 ]; then
    echo "[rank=$RANK] Running user tasks on head: ${tasks:-<none>}"
    if [[ -n "${tasks:-}" ]]; then
        bash -lc "$tasks"
    else
        echo "[rank=$RANK] No tasks specified; skipping."
    fi
    echo "[rank=$RANK] User tasks completed, starting cleanup"

    sentinel="${outdir}/cleanup_${JOB_NAME}.txt"
    echo "SHUTDOWN" > "$sentinel"

    # wait for worker ACKs (NUM_NODES includes head)
    want=$(( NUM_NODES - 1 ))
    deadline=$(( SECONDS + 180 ))
    while :; do
        have=$(ls -1 "${outdir}"/cleanup_${JOB_NAME}_ack_*.ok 2>/dev/null | wc -l | tr -d ' ')
        [ "$have" -ge "$want" ] && break
        [ $SECONDS -ge $deadline ] && { echo "[head] cleanup ack timeout ($have/$want)"; break; }
        sleep 2
    done

    # head cleanup
    do_cleanup

    # allow workers to exit now
    echo "FINALIZE" > "${outdir}/finalize_${JOB_NAME}.txt"

    # head exits (0) — OK if controller ends job now
    exit 0
else
    echo "[rank=$RANK] Worker rank; waiting for shutdown signal..."
    shutdown="${outdir}/cleanup_${JOB_NAME}.txt"
    finalize="${outdir}/finalize_${JOB_NAME}.txt"

    # wait for head’s shutdown signal
    while [ ! -f "$shutdown" ]; do sleep 5; done

    echo "[RANK ${RANK}]: Received shutdown signal."

    # ACK and run worker cleanup (don’t exit yet)
    do_cleanup
    : > "${outdir}/cleanup_${JOB_NAME}_ack_${RANK}.ok" 2>/dev/null || true

    # wait until head says it's safe to exit
    while [ ! -f "$finalize" ]; do sleep 3; done
    exit 0
fi