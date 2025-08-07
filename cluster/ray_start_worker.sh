export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export NCCL_P2P_LEVEL=NVL

while getopts ":a:c:g:t:q:" option;do
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
        object_store_memory=$q
        echo object_store_memory=$object_store_memory
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

echo "Starting ray worker @ $(hostname) with CPUs[$cpus] & GPUs [$gpus] => $cluster_address"
job="ray start --address=$cluster_address --num-cpus=$cpus --num-gpus=$gpus --temp-dir=$tmpdir --object-store-memory=$((object_store_memory))"
echo $job
$job &


if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "SLURM detected (job $SLURM_JOB_ID)"
    scheduler="slurm"
elif command -v bsub >/dev/null 2>&1; then
    echo "LSF is available on this cluster"
    lsid
    scheduler="lsf"
    echo "Ray worker LSF ID: $LSB_JOBID"
else
    echo "Neither SLURM nor LSF is available on this cluster"
    scheduler="none"
fi

sleep infinity