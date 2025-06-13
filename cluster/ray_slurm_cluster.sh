export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# parse args from `args_parser.sh` getopts
source ./args_parser.sh

tmpdir=/tmp/symlink_$(uuidgen | cut -d "-" -f5)
echo "Create symlink: $outdir -> $tmpdir"

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

export RAY_GRAFANA_HOST=${port}:3000
export RAY_PROMETHEUS_HOST=${port}:9090

############################## START HEAD NODE

nodes_array=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=(nodes_array)
head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ./ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $gpus -t $tmpdir &
sleep 10

############################## ADD WORKER NODES

worker_ids=()
num_workers=$((nodes - 1))
for i in $(seq 1 $num_workers)
do
    mkdir -p "${outdir}/ray_worker_${i}"
    echo "Adding worker: ${outdir}/ray_worker_${i}"
    if [[ "$exclusive" == "true" ]]; then
        echo "Exclusive mode is enabled"
        sbatch  --partition $partition \
                --job-name="${outdir}/ray_worker_${i}" \
                --nodes 1 \
                --ntasks 1 \
                --exclusive \
                --output="${outdir}/ray_worker_${i}.log" \
                --export=ALL \
                --wrap="apptainer exec --userns --nv \
                  --bind $workspace  --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                  $env ./ray_start_worker.sh -a $cluster_address -c $cpus -g $gpus -t $tmpdir"
    else
        sbatch  --partition $partition \
                --job-name="${outdir}/ray_worker_${i}" \
                --nodes 1 \
                --ntasks 1 \
                -n=$cpus \
                --gres=gpu:$gpus \
                --mem=$mem \
                --output="${outdir}/ray_worker_${i}.log" \
                --export=ALL \
                --wrap="apptainer exec --userns --nv \
                  --bind $workspace --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                  $env ./ray_start_worker.sh -a $cluster_address -c $cpus -g $gpus -t $tmpdir"
    fi

    jid=$(sacct -n -X --format jobid --name "${outdir}/ray_worker_${i}")
    while [ -z "$jid" ]
    do
        sleep 1
        jid=$(sacct -n -X --format jobid --name "${outdir}/ray_worker_${i}")
    done

    worker_ids+=($jid)
    echo "Running ray_worker_${i} @ ${jid}"
done

############################## CHECK STATUS

# add exit trap to ensure cleanup on script exit
# this will ensure that we stop the Ray cluster and cancel worker jobs
# even if the script fails at any point henceforth
cleanup() {
    ec=$? # exit code of the last command that triggered the trap
    echo "running cleanup (exit code: $ec)"

    # stop Ray on the head node
    apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ray stop --force

    # cancel worker jobs (if still queued/running)
    for jid in "${worker_ids[@]}"
    do
        scancel $jid
    done

    # on failure (non-zero exit) also cancel the head-node job
    [[ $ec -ne 0 ]] && scancel "$SLURM_JOB_ID" || true
}
trap cleanup EXIT

apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ./ray_check_status.sh -a $cluster_address -r $nodes

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks
