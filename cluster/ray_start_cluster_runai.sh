#!/usr/bin/env bash
set -euo pipefail

# NCCL settings optimized for Ethernet without InfiniBand
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
# export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while getopts ":i:p:d:c:g:t:q:o:" option;do
    case "${option}" in
    i)  i=${OPTARG}
        ip=$i
        echo ip=$ip
    ;;
    p)  p=${OPTARG}
        port=$p
        echo port=$port
    ;;
    d)  d=${OPTARG}
        dashboard_port=$d
        echo dashboard_port=$dashboard_port
    ;;
    c)  c=${OPTARG}
        cpus=$c
        echo cpus=$cpus
    ;;
    g)  g=${OPTARG}
        gpus=$g
        echo gpus=$gpus
    ;;
    t)  t=${OPTARG}
        tmpdir=$t
        echo tmpdir=$tmpdir
    ;;
    q)  q=${OPTARG}
        object_store_memory=$(printf "%.0f" "$q")
        echo object_store_memory=$object_store_memory
    ;;
    o)  o=${OPTARG}
        outdir=$o
        echo outdir=$outdir
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

_cleaned=0
cleanup() {
    _cleaned=1
    echo "Running head node cleanup..."
    ray stop --force >/dev/null 2>&1 || true
    echo "Successfully stopped ray head node"
    python3 /work/cell_observatory_platform/utils/cleanup.py
    echo "Successfully ran cleanup.py"
}
trap 'cleanup' EXIT
trap 'cleanup; exit 143' TERM INT

mkdir -p /tmp/ray
cluster_address="$ip:$port"

pick_agent_port() {
    local p=$((dashboard_port + 1))
    # if port lands inside the worker range, bump it out
    if (( p >= 18999 && p <= 19999 )); then
        p=$((20000 + (RANDOM % 10000)))  # 20000–29999
    fi
    echo "$p"
}
DASHBOARD_AGENT_PORT=$(pick_agent_port)

# remove any leftover shared memory segments
python3 /work/cell_observatory_platform/utils/cleanup.py

echo "Starting ray head node @ $(hostname) => $cluster_address with CPUs[$cpus] & GPUs [$gpus]"
job="ray start --block --head --node-ip-address=$ip --port=$port --dashboard-agent-listen-port=$DASHBOARD_AGENT_PORT --dashboard-host=0.0.0.0 --min-worker-port 18999 --max-worker-port 19999 --temp-dir=$tmpdir --num-cpus=$cpus --num-gpus=$gpus --object-store-memory=$object_store_memory"
echo $job
$job &
head_pid=$!

echo "$$" > "$outdir/cleanup_head.pid"

echo "[HEAD NODE]: PID for cleanup is $$"

wait "$head_pid"