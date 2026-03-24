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

hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
readarray -t hosts <<< "$hosts"
head_node=${hosts[0]}
head_node_ip=$(hostname --ip-address | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

export RAY_GRAFANA_HOST=${head_node_ip}:3000
export RAY_PROMETHEUS_HOST=${head_node_ip}:9090

########################### HELPER

do_cleanup() {
    cleanup_jobs=()

    srun -n1 -N1 -w $head_node bash -lc "
        apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $outdir:$tmpdir \
            $env bash -lc '
            pf=\"$tmpdir/cleanup_head.pid\"
            GRACE_SECONDS=60
            if [ -f \"\$pf\" ]; then
                pid=\$(cat \"\$pf\")
                kill -TERM \"\$pid\" 2>/dev/null || true
                for ((i=0;i<GRACE_SECONDS;i++)); do
                    kill -0 \"\$pid\" 2>/dev/null || break
                    sleep 1
                done
            fi
            # fallback: run cleanup ourselves
            python3 /workspace/cell_observatory_platform/utils/cleanup.py || true
            bash /workspace/cell_observatory_platform/cluster/clean_shm.sh || true
            ray stop --force >/dev/null 2>&1 || true
            '
    " >/dev/null 2>&1 &
    cleanup_jobs+=($!)

    num_workers=${#workers[@]}
    if (( num_workers > 0 )); then
        i=0
        for host in "${workers[@]}"; do
            srun -n1 -N1 -w $host bash -lc "
                apptainer exec --userns --nv \
                --bind $storage_server \
                --bind $workspace \
                --bind $bind \
                --bind $outdir/ray_worker_$i:$tmpdir \
                $env bash -lc '
                    pf=\"$tmpdir/cleanup_${i}.pid\"
                    GRACE_SECONDS=60
                    if [ -f \"\$pf\" ]; then
                        pid=\$(cat \"\$pf\")
                        kill -TERM \"\$pid\" 2>/dev/null || true
                        for ((j=0;j<GRACE_SECONDS;j++)); do
                            kill -0 \"\$pid\" 2>/dev/null || break
                            sleep 1
                        done
                    fi
                    # fallback: run cleanup ourselves
                    python3 /workspace/cell_observatory_platform/utils/cleanup.py || true
                    bash /workspace/cell_observatory_platform/cluster/clean_shm.sh || true
                    ray stop --force >/dev/null 2>&1 || true
                '
            " >/dev/null 2>&1 &
            cleanup_jobs+=($!)
            i=$((i+1))
        done
    fi

    for pid in "${cleanup_jobs[@]}"; do
        wait "$pid" || true
    done

    sleep 120

    echo "Shutting down the job"
    scancel "$SLURM_JOB_ID"
}

############################## START HEAD NODE

echo "Copying local database to head $head_node"
srun -n1 -N1 -w $head_node "
    mkdir -p /local/$USER/
    rsync -avz --stats $database_sandbox /local/$USER/sandbox.tar.gz
" >/dev/null 2>&1 &
rsync_bg_pid=$!
wait "$rsync_bg_pid"

echo "Unpacking local database on head $head_node"
srun -n1 -N1 -w $head_node "
    tar -Sxzvf /scratch/$USER/sandbox.tar.gz
" >/dev/null 2>&1 &
tar_bg_pid=$!
wait "$tar_bg_pid"

srun -n1 -N1 -w $head_node "
    apptainer exec --userns --nv \
        --bind $storage_server \
        --bind $workspace \
        --bind $bind \
        --bind $outdir:$tmpdir \
        --bind /local/$USER:/local \
        $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
        -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory
" &
head_bg_pid=$!

sleep 60
check_headnode="apptainer exec --nv \
    --bind $storage_server \
    --bind $workspace \
    --bind $bind \
    --bind $outdir:$tmpdir \
    $env ray status --address $head_node_ip:$port"
while ! $check_headnode; do
    echo "Waiting for head node..."
    sleep 3
done

############################## ADD WORKER NODES

worker_pids=()
workers=("${hosts[@]:1}")
if [ ${nodes} -gt 1 ]; then
    i=0
    for host in "${workers[@]}"; do
        echo "Copying local database to $host"
        srun -n1 -N1 -w $host "
            mkdir -p /local/$USER/
            rsync -avz --stats $database_sandbox/ /local/$USER/sandbox.tar.gz
        " >/dev/null 2>&1 &
        rsync_bg_pid=$!
        wait "$rsync_bg_pid"

        echo "Unpacking local database on $host"
        srun -n1 -N1 -w $host "
            tar -Sxzvf /local/$USER/sandbox.tar.gz
        " >/dev/null 2>&1 &
        tar_bg_pid=$!
        wait "$tar_bg_pid"

        echo "Starting worker on: $host"
        srun -n1 -N1 -w $host "
            apptainer exec --userns --nv \
                --bind $storage_server \
                --bind $workspace \
                --bind $bind \
                --bind $outdir/ray_worker_$i:$tmpdir \
                --bind /local/$USER:/local \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i
        " &
        worker_pids+=($!)
        i+=1
    done
fi

############################## RUN WORKLOAD

# trap 'do_cleanup' EXIT
trap 'do_cleanup; exit 130' INT # SIGINT
trap 'do_cleanup; exit 143' TERM # SIGTERM like bkill
trap 'do_cleanup; exit 140' TERM # TERM_RUNLIMIT 

# CHECK CLUSTER STATUS
srun -n1 -N1 -w $head_node " 
    apptainer exec --userns --nv \
        --bind $storage_server \
        --bind $workspace \
        --bind $bind \
        --bind $outdir:$tmpdir \
        --bind /local/$USER:/local \
        $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
        -a $cluster_address -r $nodes 
"
rc=$?
if [ $rc -ne 0 ]; then
    echo "Cluster failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv \
    --bind $storage_server \
    --bind $workspace \
    --bind $bind \
    --bind $outdir:$tmpdir \
    --bind /local/$USER:/local \
    $env $tasks


############################## CLEANUP

echo "User tasks completed, starting cleanup"
do_cleanup
exit 0