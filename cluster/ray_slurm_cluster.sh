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

hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
hosts=(hosts)
head_node=${hosts[0]}
head_node_ip=$(hostname --ip-address | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

srun -n1 -N1 -w $head_node "
    apptainer exec --userns --nv \
        --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
        $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
        -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory
" &
sleep 10
check_headnode="ray status --address $head_node:\$port"
while ! $check_headnode; do
    echo "Waiting for head node..."
    sleep 3
done

############################## ADD WORKER NODES

workers=("${hosts[@]:1}")
if [ ${nodes} -gt 1 ]; then
    i=0
    for host in "${workers[@]}"; do
        echo "Starting worker on: $host"
        srun -n1 -N1 -w $host "
            apptainer exec --userns --nv \
                --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_$i:$tmpdir \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory
        " &
        i+=1
    done
fi

############################# CHECK CLUSTER STATUS

srun -n1 -N1 -w $head_node " 
    apptainer exec --userns --nv \
        --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
        $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
        -a $cluster_address -r $nodes 
" &

############################## CLEANUP

# add exit trap to ensure cleanup on script exit
# this will ensure that we stop the Ray cluster and cancel worker jobs
# even if the script fails at any point henceforth
cleanup() {
    ec=$? # exit code of the last command that triggered the trap
    echo "running cleanup (exit code: $ec)"

    # stop Ray on the head node
    ps aux | grep prometheus | awk '{print $2}' | xargs kill -9

    for host in "${hosts[@]}"; do
        srun -n1 -N1 -w $host " 
            apptainer exec --userns --nv \
                --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
                $env ray stop --force 
        " &
    done
    wait

    # on failure (non-zero exit) also cancel the head-node job
    [[ $ec -ne 0 ]] && scancel "$SLURM_JOB_ID" || true
}
trap cleanup EXIT

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks
