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

head_node=$(cat $LSB_DJOB_HOSTFILE | uniq | head -n1 | awk '{print $1;}')
head_node_ip=$(getent hosts $head_node | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory &

############################## ADD WORKER NODES

worker_ids=()
num_workers=$((nodes - 1))
for i in $(seq 1 $num_workers)
do
    mkdir -p "${outdir}/ray_worker_${i}"
    echo "Adding worker: ${outdir}/ray_worker_${i}"
    if [[ "$exclusive" == "true" ]]; then
        echo "Exclusive mode is enabled"
        job="bsub -cwd "$(pwd)" \
            -q $partition \
            -J "${jobname}_ray_worker_${i}" \
            -x \
            -n $cpus \
            -gpu "num=$gpus:mode=shared" \
            -o "${outdir}/ray_worker_${i}.log" \
            apptainer exec --userns --nv \
              --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory"
    else
        job="bsub -cwd "$(pwd)" \
            -q $partition \
            -J "${jobname}_ray_worker_${i}" \
            -n $cpus \
            -gpu "num=$gpus:mode=shared" \
            -o "${outdir}/ray_worker_${i}.log" \
            apptainer exec --userns --nv \
              --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory"
    fi

    echo $job
    $job


    jid=$(bjobs -r -J "${jobname}_ray_worker_${i}" | awk 'NR==2 {print $1;}')
    while [ -z "$jid" ]
    do
        sleep 1
        jid=$(bjobs -r -J "${jobname}_ray_worker_${i}" | awk 'NR==2 {print $1;}')
    done

    worker_ids+=($jid)
    echo "Running ${jobname}_ray_worker_${i} @ ${jid}"
done

############################# CHECK STATUS

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh -a $cluster_address -r $nodes

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks

############################## CLEANUP

echo "Stop ray"
ps aux | grep prometheus | awk '{print $2}' | xargs kill -9
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ray stop --force

echo "Shutting down the Job"

for jid in "${worker_ids[@]}"
do
    bkill $jid
done

bkill $LSB_JOBID