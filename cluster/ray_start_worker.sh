export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export NCCL_P2P_LEVEL=NVL

while getopts ":a:c:g:t:" option;do
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
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

echo "Starting ray worker @ $(hostname) with CPUs[$cpus] & GPUs [$gpus] => $cluster_address"
job="ray start --address=$cluster_address --num-cpus=$cpus --num-gpus=$gpus --temp-dir=$tmpdir"
echo $job
$job &


if command -v srun >/dev/null 2>&1; then
    echo "SLURM is available on this cluster"
    srun --version
    scheduler="slurm"
    echo "Ray worker SLURM ID: $SLURM_JOB_ID"

elif command -v bsub >/dev/null 2>&1; then
    echo "LSF is available on this cluster"
    lsid
    scheduler="lsf"
else
    echo "Neither SLURM nor LSF is available on this cluster"
    scheduler="none"
    echo "Ray worker LSF ID: $LSB_JOBID"
fi

sleep infinity