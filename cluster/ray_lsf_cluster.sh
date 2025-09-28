#!/usr/bin/env bash

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

# Get allocated hosts from LSF
hosts=()
for host in $(cat $LSB_DJOB_HOSTFILE | uniq); do
    echo "Adding host: $host"
    hosts+=($host)
done
echo "The host list is: ${hosts[@]}"

head_node=${hosts[0]}
head_node_ip=$(getent hosts $head_node | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

blaunch -z $head_node "
    apptainer exec --userns --nv \
        --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
        $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
        -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory
" &
head_bg_pid=$!

sleep 10

apptainer exec --userns --nv \
    --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
    $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
    -a $cluster_address -r 1


############################## ADD WORKER NODES

worker_pids=()
workers=("${hosts[@]:1}")
if [ ${nodes} -gt 1 ]; then
    i=0
    for host in "${workers[@]}"; do
        echo "Starting worker on: $host"
        mkdir -p $outdir/ray_worker_$i
        blaunch -z $host "
            apptainer exec --userns --nv \
                --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_$i:$tmpdir \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i
        " &
        worker_pids+=($!)
        i+=1
    done
fi

############################# CHECK CLUSTER STATUS

blaunch -z $head_node " 
    apptainer exec --userns --nv \
        --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
        $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
        -a $cluster_address -r $nodes 
"

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks

############################## CLEANUP

blaunch -z $head_node bash -lc "
    apptainer exec --userns --nv \
        --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir \
        $env bash -lc '
        pf=\"$tmpdir/cleanup_head.pid\"
        GRACE_SECONDS=20
        if [ -f \"\$pf\" ]; then
            pid=\$(cat \"\$pf\")
            kill -TERM \"\$pid\" 2>/dev/null || true
            for ((i=0;i<GRACE_SECONDS;i++)); do
                kill -0 \"\$pid\" 2>/dev/null || exit 0
                sleep 1
            done
            kill -KILL \"\$pid\" 2>/dev/null || true
        fi
        '
    " >/dev/null 2>&1 &

num_workers=${#workers[@]}
if (( num_workers > 0 )); then
    i=0
    for host in "${workers[@]}"; do
        blaunch -z $host bash -lc "
            apptainer exec --userns --nv \
            --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_$i:$tmpdir \
            $env bash -lc '
                pf=\"$tmpdir/cleanup_${i}.pid\"
                GRACE_SECONDS=20
                if [ -f \"\$pf\" ]; then
                    pid=\$(cat \"\$pf\")
                    kill -TERM \"\$pid\" 2>/dev/null || true
                    for ((j=0;j<GRACE_SECONDS;j++)); do
                        kill -0 \"\$pid\" 2>/dev/null || exit 0
                        sleep 1
                    done
                    kill -KILL \"\$pid\" 2>/dev/null || true
                fi
            '
        " >/dev/null 2>&1 &
        i=$((i+1))
    done
fi

kill -KILL $head_bg_pid 2>/dev/null || true
for pid in "${worker_pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
done

wait || true

echo "Shutting down the job"
bkill $LSB_JOBID