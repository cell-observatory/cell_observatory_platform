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

############################## JOB CHAINING (clusters.chain_jobs > 1)
# Exits immediately if the run is already complete; otherwise submits the
# follower job (held until this one ends) and records this job's exit.
source "$DIR/chain_lib.sh"
chain_job_start "$outdir"

tmpdir=/tmp/symlink_$(uuidgen | cut -d "-" -f5)
echo "Create symlink: $outdir -> $tmpdir"

# Node-local scratch: default to /scratch/$USER (Janelia convention).
# If the submit wrapper already exports SCRATCH_ROOT or NODE_LOCAL_STORE_ROOT,
# those explicit values win. Otherwise derive everything from /scratch/$USER.
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
NODE_LOCAL_STORE_ROOT="${SCRATCH_ROOT}/node_local_store"
mkdir -p "$NODE_LOCAL_STORE_ROOT"
export SCRATCH_ROOT NODE_LOCAL_STORE_ROOT
echo "Using SCRATCH_ROOT: $SCRATCH_ROOT"
echo "Using NODE_LOCAL_STORE_ROOT: $NODE_LOCAL_STORE_ROOT"

# Segregate ephemeral /scratch (wiped by cleanup.py) from the postgres sandbox
# (which must persist for the lifetime of the job):
#   training_scratch -> bound to /scratch in ray containers; safe to nuke
#   sandbox_dir      -> postgres rootfs (apptainer instance "mysql"); NOT
#                       bound to /scratch, so cleanup.py can't reach it
training_scratch="${SCRATCH_ROOT}/training"
sandbox_dir="${SCRATCH_ROOT}/pgdb/sandbox"
sandbox_tar="${SCRATCH_ROOT}/pgdb/sandbox.tar.zst"

: "${SUPABASE_LOCAL_PORT:?SUPABASE_LOCAL_PORT must be set in the environment (see .env)}"

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

############################## PER-NODE HELPERS
#
# Every helper that touches a remote node does so via
#     blaunch -z "$host" "..."
# By centralising these here we guarantee the head and every worker get the
# same recipe (DB staging, postgres start, ray bring-up, cleanup), so any
# cluster-specific workaround only needs to be applied once.

# stage_local_db_on_node <host>
#   Sync sandbox.tar.zst to <host>'s scratch root and unpack it under
#   $SCRATCH_ROOT/pgdb/, producing $sandbox_dir as the postgres rootfs.
#   Also pre-creates $training_scratch and $NODE_LOCAL_STORE_ROOT so
#   subsequent --bind mounts don't fail. Synchronous.
stage_local_db_on_node() {
    local host=$1

    echo "Copying local database to $host"
    blaunch -z "$host" "
        mkdir -p $SCRATCH_ROOT/pgdb $training_scratch $NODE_LOCAL_STORE_ROOT
        rsync -avz --stats $database_sandbox $sandbox_tar
    "

    echo "Unpacking local database on $host"
    blaunch -z "$host" "
        zstd -dc $sandbox_tar | tar -xf - -C $SCRATCH_ROOT/pgdb
    "
}

# start_local_db_on_node <host> <out_pid_var>
#   Start the postgres apptainer instance on <host> and run postgres in the
#   foreground inside the same blaunch step. Backgrounded; <out_pid_var>
#   receives the blaunch bg pid.
start_local_db_on_node() {
    local host=$1
    local -n out_pid=$2

    echo "Starting local database on $host"
    blaunch -z "$host" "
        apptainer instance stop mysql >/dev/null 2>&1 || true
        # Defensive scrub: if a previous job died before pg_ctl stop could
        # run, postmaster.pid will be left in the sandbox and the next
        # postgres start will FATAL. Only safe to delete iff no postgres
        # process is still alive on this node holding our port. Postgres
        # is launched with '-c port=<N>', so match on that exact token.
        if ! pgrep -f \"postgres .* port=$SUPABASE_LOCAL_PORT\" >/dev/null 2>&1; then
            rm -f $sandbox_dir/var/lib/postgresql/data/postmaster.pid
        fi
        mkdir -p $sandbox_dir/global \
                 $sandbox_dir/clusterfs \
                 $sandbox_dir/scratch \
                 $sandbox_dir/dev/shm \
                 $sandbox_dir/groups
        env -u APPTAINER_BIND -u APPTAINER_BINDPATH \
            -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
            apptainer instance start --no-mount proc --writable \
                --env POSTGRES_PASSWORD=postgres \
                $sandbox_dir mysql
        apptainer exec instance://mysql postgres \
            -c 'listen_addresses=0.0.0.0' \
            -c 'port=$SUPABASE_LOCAL_PORT' \
            -c 'config_file=/etc/postgresql/postgresql.conf'
    " &
    out_pid=$!
}

# stop_local_db_on_node <host> <out_pid_array>
#   Gracefully shut down postgres inside the running mysql instance, then
#   stop the apptainer instance itself. Without the pg_ctl step, `apptainer
#   instance stop` only signals the instance's main process (the sandbox
#   runscript); the exec'd postgres is reaped by namespace teardown, leaving
#   postmaster.pid orphaned in the sandbox and blocking the next startup.
stop_local_db_on_node() {
    local host=$1
    local -n out_pids=$2

    blaunch -z "$host" "
        apptainer exec instance://mysql pg_ctl -D /var/lib/postgresql/data -m fast stop -w -t 30 \
            >/dev/null 2>&1 || true
        apptainer instance stop mysql >/dev/null 2>&1 || true
    " >/dev/null 2>&1 &
    out_pids+=($!)
}

# wait_for_local_db_on_node <host> <host_ip>
#   Block until pg_isready (run inside the mysql apptainer instance on
#   <host>) reports the DB is accepting TCP connections, or fail after ~120s.
wait_for_local_db_on_node() {
    local host=$1
    local host_ip=$2

    echo "Waiting for local database on $host to be ready"
    blaunch -z "$host" "
        for ((attempt=1; attempt<=60; attempt++)); do
            if apptainer exec instance://mysql pg_isready -h $host_ip -p $SUPABASE_LOCAL_PORT -d postgres >/dev/null 2>&1; then
                echo \"Local DB on $host ready (attempt \$attempt)\"
                exit 0
            fi
            echo \"Waiting for local DB on $host (attempt \$attempt/60)\"
            sleep 2
        done
        echo \"Local DB on $host failed readiness check\" >&2
        exit 1
    "
}

# start_ray_head_on_node <host> <out_pid_var>
start_ray_head_on_node() {
    local host=$1
    local -n out_pid=$2

    blaunch -z "$host" "
        apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $outdir:$tmpdir \
            --bind $training_scratch:/scratch \
            --bind $NODE_LOCAL_STORE_ROOT:$NODE_LOCAL_STORE_ROOT \
            $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh \
            -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus -t $tmpdir -q $object_store_memory
    " &
    out_pid=$!
}

# start_ray_worker_on_node <host> <worker_id> <out_pid_var>
start_ray_worker_on_node() {
    local host=$1
    local worker_id=$2
    local -n out_pid=$3

    echo "Starting worker on: $host"
    mkdir -p $outdir/ray_worker_$worker_id
    blaunch -z "$host" "
        apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $outdir/ray_worker_$worker_id:$tmpdir \
            --bind $training_scratch:/scratch \
            --bind $NODE_LOCAL_STORE_ROOT:$NODE_LOCAL_STORE_ROOT \
            $env /workspace/cell_observatory_platform/cluster/ray_start_worker.sh \
            -a $cluster_address -c $cpus -g $gpus -t $tmpdir -q $object_store_memory -w $worker_id
    " &
    out_pid=$!
}

# wait_for_ray_cluster <required_nodes>
wait_for_ray_cluster() {
    local required=$1

    blaunch -z "$head_node" "
        apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $outdir:$tmpdir \
            --bind $training_scratch:/scratch \
            $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh \
            -a $cluster_address -r $required
    "
}

# run_user_task
run_user_task() {
    echo "Running user tasks"
    echo "$tasks"
    apptainer exec --userns --nv \
        --bind $storage_server \
        --bind $workspace \
        --bind $bind \
        --bind $outdir:$tmpdir \
        --bind $training_scratch:/scratch \
        --bind $NODE_LOCAL_STORE_ROOT:$NODE_LOCAL_STORE_ROOT \
        $env $tasks
}

# cleanup_node <host> <tmpdir_bindsrc> <pidfile_basename> <out_pid_array>
#   Single responsibility: deliver SIGTERM to the head/worker pid written
#   into the bind-mounted tmpdir and wait for that process's own EXIT trap
#   (defined in ray_start_cluster.sh / ray_start_worker.sh) to run ray stop
#   and the per-node scratch/shm scrub. After the grace window, SIGKILL.
#
#   We deliberately do NOT spawn a fresh `apptainer exec` here to re-run
#   `ray stop`, `cleanup.py`, or `clean_shm.sh`:
#     - `ray stop` from a separate container has no view of the original
#       ray runtime sockets, so it would be a no-op.
#     - cleanup.py and clean_shm.sh are already owned by the trapped cleanup
#       inside the worker/head container (and by LSF `-Ep clean_shm.sh` as
#       a post-exec belt-and-suspenders).
#
#   Postgres / apptainer instance shutdown is owned by stop_local_db_on_node.
cleanup_node() {
    local host=$1
    local tmpdir_bindsrc=$2
    local pidfile=$3
    local -n out_pids=$4

    blaunch -z "$host" "
        pf=\"$tmpdir_bindsrc/$pidfile\"
        GRACE_SECONDS=60
        if [ -f \"\$pf\" ]; then
            pid=\$(cat \"\$pf\")
            kill -TERM \"\$pid\" 2>/dev/null || true
            for ((i=0;i<GRACE_SECONDS;i++)); do
                kill -0 \"\$pid\" 2>/dev/null || break
                sleep 1
            done
            kill -KILL \"\$pid\" 2>/dev/null || true
        fi
    " >/dev/null 2>&1 &
    out_pids+=($!)
}

# do_cleanup
#   Two-phase fan-out:
#     1. Signal head + every worker (in parallel) and wait for their trapped
#        cleanups to drain.
#     2. Gracefully stop postgres + apptainer instance on every node.
#   Idempotent via _ORCH_CLEANED guard so re-entry from a second TERM (or
#   the LSF runlimit) does not double-fan-out and race the first round of
#   blaunches into "Failed while waiting for tasks to finish".
_ORCH_CLEANED=0
do_cleanup() {
    (( _ORCH_CLEANED )) && return
    _ORCH_CLEANED=1

    local out_pids=()
    cleanup_node "$head_node" "$outdir" "cleanup_head.pid" out_pids
    local i=0
    for host in "${workers[@]:-}"; do
        cleanup_node "$host" "$outdir/ray_worker_$i" "cleanup_${i}.pid" out_pids
        i=$((i+1))
    done
    for pid in "${out_pids[@]}"; do wait "$pid" || true; done

    out_pids=()
    stop_local_db_on_node "$head_node" out_pids
    for host in "${workers[@]:-}"; do
        stop_local_db_on_node "$host" out_pids
    done
    for pid in "${out_pids[@]}"; do wait "$pid" || true; done

    echo "Cleanup complete"

    # Buffer for any straggling blaunch tear-downs, then hard-kill the LSF
    # allocation. Without this, if anything in the orchestrator shell aborts
    # (e.g. parse error from editing this script mid-run) the workers'
    # backgrounded blaunches can keep the allocation in RUN until runlimit.
    sleep 120
    echo "Shutting down the job"
    bkill "$LSB_JOBID"
}

############################## START HEAD NODE

stage_local_db_on_node "$head_node"
start_local_db_on_node "$head_node" db_head_bg_pid

# Deliberately do NOT export SUPABASE_LOCAL_HOST.
# data/databases/local_database.py constructs its connection URI from
# utils.context.node_ip(), which asks ray for the calling actor's
# NodeManagerAddress. We start postgres on every node, so each actor
# should resolve its OWN node's IP rather than all pointing at head.

start_ray_head_on_node "$head_node" head_bg_pid

sleep 60

# Fail fast if the postgres blaunch already died (e.g. apptainer instance
# start couldn't satisfy a system bind path).
if ! kill -0 "$db_head_bg_pid" 2>/dev/null; then
    wait "$db_head_bg_pid"
    db_rc=$?
    echo "Local DB step on $head_node exited early with code $db_rc; aborting" >&2
    do_cleanup
    exit "$db_rc"
fi

# Initial ray-up loop: cheap `ray status` from the orchestrator shell until
# the head responds, before spinning up workers.
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
db_worker_pids=()
worker_ips=()
workers=("${hosts[@]:1}")
i=0
for host in "${workers[@]}"; do
    worker_ip=$(getent hosts "$host" | awk '{ print $1 }')
    worker_ips+=("$worker_ip")

    stage_local_db_on_node "$host"

    start_local_db_on_node "$host" worker_db_pid
    db_worker_pids+=("$worker_db_pid")

    start_ray_worker_on_node "$host" "$i" worker_pid
    worker_pids+=("$worker_pid")

    i=$((i+1))
done

############################## RUN WORKLOAD

trap 'do_cleanup; exit 130' INT  # SIGINT
trap 'do_cleanup; exit 143' TERM # SIGTERM / TERM_RUNLIMIT

# Multi-node ray cluster status gate.
wait_for_ray_cluster "$nodes"
rc=$?
if [ $rc -ne 0 ]; then
    echo "Cluster failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

# DB readiness gate, fanned out across head + every worker in parallel.
# Each ray actor queries the postgres on its OWN node, so all must be up.
db_check_pids=()
db_check_hosts=("$head_node" "${workers[@]}")
db_check_ips=("$head_node_ip" "${worker_ips[@]}")

for idx in "${!db_check_hosts[@]}"; do
    host=${db_check_hosts[$idx]}
    host_ip=${db_check_ips[$idx]}
    ( wait_for_local_db_on_node "$host" "$host_ip" ) &
    db_check_pids+=("$!:$host")
done

db_failed=0
for entry in "${db_check_pids[@]}"; do
    pid=${entry%%:*}
    host=${entry#*:}
    if ! wait "$pid"; then
        echo "Local DB readiness check failed on $host" >&2
        db_failed=1
    fi
done
if [ "$db_failed" -ne 0 ]; then
    echo "One or more local DBs failed readiness check, exiting" >&2
    do_cleanup
    exit 1
fi

run_user_task

############################## CLEANUP

echo "User tasks completed, starting cleanup"
do_cleanup
exit 0
