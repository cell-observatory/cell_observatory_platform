#!/usr/bin/env bash

export NCCL_DEBUG=INFO
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

nodes_array=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
readarray -t nodes_array <<< "$nodes_array"
head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind /dev/shm:/dev/shm \
    --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
    -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory &

sleep 60

check_headnode="apptainer exec --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ray status --address $head_node_ip:$port"
while ! $check_headnode; do
    echo "Waiting for head node..."
    sleep 3
done

############################## ADD WORKER NODES

worker_ids=()
workers=("${nodes_array[@]:1}")
num_workers=$((nodes - 1))
for i in $(seq 1 $num_workers)
do
    mkdir -p "${outdir}/ray_worker_${i}"
    echo "Adding worker: ${outdir}/ray_worker_${i}"
    if [[ "$exclusive" == "true" ]]; then
        echo "Exclusive mode is enabled"
        jid=$(sbatch --parsable --partition $partition \
                --job-name="${jobname}_ray_worker_${i}" \
                --nodes 1 \
                --ntasks 1 \
                --exclusive \
                --output="${outdir}/ray_worker_${i}.log" \
                --export=ALL \
                --wrap="apptainer exec --userns --nv \
                  --bind $storage_server --bind $workspace  --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                  $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                  -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i" \
                | awk '{print $4}')
    else
        jid=$(sbatch --parsable --partition $partition \
                --job-name="${jobname}_ray_worker_${i}" \
                --nodes 1 \
                --ntasks 1 \
                --cpus-per-task=$cpus \
                --gres=gpu:$gpus \
                --mem=$mem \
                --output="${outdir}/ray_worker_${i}.log" \
                --export=ALL \
                --wrap="apptainer exec --userns --nv \
                  --bind $storage_server --bind $workspace --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir \
                  $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                  -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i" \
                | awk '{print $4}')
    fi

    worker_ids+=($jid)
    echo "Running ${jobname}_ray_worker_${i} @ ${jid}"
done

############################## CHECK STATUS

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind \
    --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh -a $cluster_address -r $nodes

############################## RUN WORKLOAD

# FIXME: (IMPORTANT!) we need to add a trap here to ensure cleanup on exit/signals

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks

############################## CLEANUP

wait_for_jobs() {
    local -i timeout="$1"; shift
    (( $# == 0 )) && return 0

    local -a jids=("$@")
    local -i t=0
    while (( t < timeout )); do
        local any=0
        for jid in "${jids[@]}"; do
            squeue -h -j "$jid" >/dev/null 2>&1 && any=1
        done
        (( any == 0 )) && return 0
        sleep 1; ((t++))
    done
    return 1
}

head_pid=$(cat "$outdir/cleanup_head.pid" 2>/dev/null || true)
if [[ -n "$head_pid" ]]; then
    kill -TERM "$head_pid" 2>/dev/null || true
    for _ in {1..120}; do
        kill -0 "$head_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$head_pid" 2>/dev/null || true
fi

# TODO: requires further multinode testing
if (( ${#worker_ids[@]} )); then
    for jid in "${worker_ids[@]}"; do
        scancel --signal=TERM "$jid" 2>/dev/null || true
    done
    if ! wait_for_jobs 120 "${worker_ids[@]}"; then
        for jid in "${worker_ids[@]}"; do
            scancel "$jid" 2>/dev/null || true
        done
    fi
fi

echo "Shutting down the job"
scancel "$SLURM_JOB_ID"