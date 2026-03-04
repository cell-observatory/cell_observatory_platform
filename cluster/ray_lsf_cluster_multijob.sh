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
export USER_POSTEXEC=/workspace/cell_observatory_platform/cluster/clean_shm.sh


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

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind \
    --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
    -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory &
head_bg_pid=$!

sleep 60

check_headnode="apptainer exec --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ray status --address $head_node_ip:$port"
while ! $check_headnode; do
    echo "Waiting for head node..."
    sleep 3
done

############################## ADD WORKER NODES

worker_ids=()
num_workers=$((nodes - 1))
for i in $(seq 1 $num_workers)
do
    mkdir -p "${outdir}/ray_worker_${i}"
    echo "Adding worker: ${outdir}/ray_worker_${i}"
    if [[ "$exclusive" == "true" ]]; then
        echo "Exclusive mode is enabled"
        bsub -cwd "$(pwd)" \
            -q $partition \
            -J "${jobname}_ray_worker_${i}" \
            -x \
            -n $cpus \
            -gpu "num=$gpus:mode=shared" \
            -o "${outdir}/ray_worker_${i}.log" \
            apptainer exec --userns --nv \
              --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i
    else
        bsub -cwd "$(pwd)" \
            -q $partition \
            -J "${jobname}_ray_worker_${i}" \
            -n $cpus \
            -gpu "num=$gpus:mode=shared" \
            -o "${outdir}/ray_worker_${i}.log" \
            apptainer exec --userns --nv \
              --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i
    fi

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

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind \
    --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh -a $cluster_address -r $nodes

############################## RUN WORKLOAD

# FIXME: (IMPORTANT!) we need to add a trap here to ensure cleanup on exit/signals

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks

############################## CLEANUP

wait_for_lsf_jobs() {
    local -i timeout="$1"; shift
    (( $# == 0 )) && return 0

    local -a jids=("$@")
    local -i t=0
    while (( t < timeout )); do
        local any=0
        for jid in "${jids[@]}"; do
            if bjobs -noheader "$jid" >/dev/null 2>&1; then
                any=1
            fi
        done
        (( any == 0 )) && return 0
        sleep 1; ((t++))
    done
    return 1
}

head_pid=$(cat "$outdir/cleanup_head.pid" 2>/dev/null || true)
if [[ -n "$head_pid" ]]; then
    kill -TERM "$head_pid" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 "$head_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$head_pid" 2>/dev/null || true
fi

if (( ${#worker_ids[@]} )); then
    for jid in "${worker_ids[@]}"; do
        bkill -s TERM "$jid" 2>/dev/null || true
    done
    if ! wait_for_lsf_jobs 30 "${worker_ids[@]}"; then
        for jid in "${worker_ids[@]}"; do
            bkill "$jid" 2>/dev/null || true
        done
    fi
fi

echo "Shutting down the job"
bkill $LSB_JOBID