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
export RAY_GRAFANA_HOST="${head_node_ip}:3000"
export RAY_PROMETHEUS_HOST="${head_node_ip}:9090"

: "${SUPABASE_LOCAL_PORT:?SUPABASE_LOCAL_PORT must be set in the environment}"

########################### HELPER

start_local_db() {
    local target_host=$1
    local host_ip=$2
    echo "Starting local database on $target_host"
    blaunch -z "$target_host" "
        apptainer instance stop mysql >/dev/null 2>&1 || true
        apptainer instance start --no-mount proc --writable --env POSTGRES_PASSWORD=postgres /scratch/$USER/sandbox mysql
        apptainer exec instance://mysql postgres \
            -c 'listen_addresses=0.0.0.0' \
            -c 'port=${SUPABASE_LOCAL_PORT}' \
            -c 'config_file=/etc/postgresql/postgresql.conf' &
    "
    local uri="postgresql://postgres:postgres@${host_ip}:${SUPABASE_LOCAL_PORT}/postgres"
    local attempts=30
    for ((attempt=1; attempt<=attempts; attempt++)); do
        if blaunch -z "$target_host" "pg_isready -h $host_ip -p ${SUPABASE_LOCAL_PORT} -d postgres" >/dev/null 2>&1; then
            echo "Local database on $target_host is ready"
            return 0
        fi
        echo "Waiting for local database on $target_host (attempt ${attempt}/${attempts})"
        sleep 2
    done
    echo "Local database on $target_host failed readiness check" >&2
    return 1
}

do_cleanup() {
    cleanup_jobs=()

    blaunch -z $head_node "
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
        apptainer instance stop mysql >/dev/null 2>&1 || true
    " >/dev/null 2>&1 &
    cleanup_jobs+=($!)

    num_workers=${#workers[@]}
    if (( num_workers > 0 )); then
        i=0
        for host in "${workers[@]}"; do
            blaunch -z $host "
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
                apptainer instance stop mysql >/dev/null 2>&1 || true
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
    bkill $LSB_JOBID
}

############################## START HEAD NODE

echo "Copying local database to head $head_node"
blaunch -z $head_node "
    mkdir -p /scratch/$USER/
    rsync -avz --stats $database_sandbox /scratch/$USER/sandbox.tar.zst
" >/dev/null 2>&1 &
rsync_bg_pid=$!
wait "$rsync_bg_pid"

echo "Unpacking local database on head $head_node"
blaunch -z $head_node "
    tar --zstd -xf /scratch/$USER/sandbox.tar.zst -C /scratch/$USER
" >/dev/null 2>&1 &
tar_bg_pid=$!
wait "$tar_bg_pid"

start_local_db "$head_node" "$head_node_ip"
export SUPABASE_LOCAL_HOST="$head_node_ip"
export SUPABASE_LOCAL_URI="postgresql://postgres:postgres@${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}/postgres"

blaunch -z $head_node "
    apptainer exec --userns --nv \
        --bind $storage_server \
        --bind $workspace \
        --bind $bind \
        --bind $outdir:$tmpdir \
        --bind /scratch/$USER:/scratch \
        $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
            -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory
" &
head_bg_pid=$!

sleep 60

apptainer exec --userns --nv \
    --bind $storage_server \
    --bind $workspace \
    --bind $bind \
    --bind $outdir:$tmpdir \
    $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
    -a $cluster_address -r 1
rc=$?
if [ $rc -ne 0 ]; then
    echo "Head node failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

############################## ADD WORKER NODES

worker_pids=()
workers=("${hosts[@]:1}")
if [ ${nodes} -gt 1 ]; then
    i=0
    for host in "${workers[@]}"; do
        echo "Copying local database to $host"
        blaunch -z $host "
            mkdir -p /scratch/$USER/
            rsync -avz --stats $database_sandbox /scratch/$USER/sandbox.tar.zst
        " >/dev/null 2>&1 &
        rsync_bg_pid=$!
        wait "$rsync_bg_pid"

        echo "Unpacking local database on $host"
        blaunch -z $host "
            tar --zstd -xf /scratch/$USER/sandbox.tar.zst -C /scratch/$USER
        " >/dev/null 2>&1 &
        tar_bg_pid=$!
        wait "$tar_bg_pid"

        worker_ip=$(getent hosts $host | awk '{ print $1 }')
        start_local_db "$host" "$worker_ip"

        echo "Starting worker on: $host"
        mkdir -p $outdir/ray_worker_$i
        blaunch -z "$host" "
        apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $outdir/ray_worker_$i:$tmpdir \
            --bind /scratch/$USER:/scratch \
            $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
                -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $i
        " &
        worker_pids+=($!)
        i+=1
    done
fi

############################## RUN WORKLOAD

trap 'do_cleanup; exit 130' INT # SIGINT
trap 'do_cleanup; exit 143' TERM # SIGTERM / TERM_RUNLIMIT

# CHECK CLUSTER STATUS
blaunch -z $head_node " 
    apptainer exec --userns --nv \
        --bind $storage_server \
        --bind $workspace \
        --bind $bind \
        --bind $outdir:$tmpdir \
        --bind /scratch/$USER:/scratch \
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
    --bind /scratch/$USER:/scratch \
    $env $tasks

############################## CLEANUP

echo "User tasks completed, starting cleanup"
do_cleanup
exit 0