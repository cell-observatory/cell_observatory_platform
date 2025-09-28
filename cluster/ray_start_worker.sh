#!/usr/bin/env bash
set -Eeuo pipefail

# NCCL settings optimized for Ethernet without InfiniBand
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while getopts ":a:c:g:t:q:w:" option;do
    case "${option}" in
    a)  a=${OPTARG}
        cluster_address=$a
        echo cluster_address=$cluster_address
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
    w)  w=${OPTARG}
        worker_id=$w
        echo worker_id=$worker_id
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

_cleaned=0
cleanup() {
    (( _cleaned )) && return
    _cleaned=1
    echo "Running worker node cleanup..."
    ray stop --force >/dev/null 2>&1 || true
    python3 /workspace/cell_observatory_platform/utils/cleanup.py || true
}
trap 'cleanup' EXIT
trap 'cleanup; exit 143' TERM INT USR1

echo "Starting ray worker @ $(hostname) with CPUs[$cpus] & GPUs [$gpus] => $cluster_address"
job="ray start --block --address=$cluster_address --num-cpus=$cpus --num-gpus=$gpus --temp-dir=$tmpdir --object-store-memory=$object_store_memory"
$job &
ray_pid=$!

if [[ -n "${worker_id:-}" ]]; then
  echo "$$" > "$tmpdir/cleanup_${worker_id}.pid"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM detected (job ${SLURM_JOB_ID})"
    scheduler="slurm"
elif command -v bsub >/dev/null 2>&1; then
    echo "LSF is available on this cluster"
    lsid || true
    scheduler="lsf"
    # echo "Ray worker LSF ID: ${LSB_JOBID:-N/A}"
else
    echo "Neither SLURM nor LSF is available on this cluster"
    scheduler="none"
fi

wait "$ray_pid"