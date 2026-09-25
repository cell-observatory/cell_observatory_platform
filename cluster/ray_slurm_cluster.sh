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

# Set SCRATCH_ROOT to a user-writable, node-local scratch directory.
# If running under Slurm, prefer SLURM_TMPDIR if defined (some clusters provide this as per-job tmp space),
# else fall back to TMPDIR if defined (e.g., some Slurm setups or PrologFlags=Contain). If neither is set,
# use /tmp/$USER as a last resort. Always scope scratch by user to avoid collisions.
# NODE_LOCAL_STORE_ROOT is placed within SCRATCH_ROOT for consistency with other ray_*_cluster.sh scripts.
SCRATCH_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}"
NODE_LOCAL_STORE_ROOT="${SCRATCH_ROOT}/node_local_store"
mkdir -p "$NODE_LOCAL_STORE_ROOT"
export SCRATCH_ROOT NODE_LOCAL_STORE_ROOT
echo "Using SCRATCH_ROOT: $SCRATCH_ROOT"
echo "Using NODE_LOCAL_STORE_ROOT: $NODE_LOCAL_STORE_ROOT"

# Segregate the ephemeral /scratch (which utils/cleanup.py wipes on every
# ray head/worker bring-up via clean_scratch_directory()) from the postgres
# sandbox (which must persist for the lifetime of the job):
#   training_scratch -> bound to /scratch in ray containers; safe to nuke
#   sandbox_dir      -> postgres rootfs (apptainer instance "mysql"); NOT
#                       bound to /scratch, so cleanup.py can't reach it
# Without this split, cleanup.py rmtree's the running PGDATA and postgres
# self-shuts-down ~60s later when its postmaster.pid heartbeat fails.
training_scratch="${SCRATCH_ROOT}/training"
sandbox_dir="${SCRATCH_ROOT}/pgdb/sandbox"
sandbox_tar="${SCRATCH_ROOT}/pgdb/sandbox.tar.zst"

# Local Postgres (apptainer instance "mysql") TCP port. Required by the
# LocalArrowDatabase clients launched inside ray train workers; if it's not
# exported the workers fall back to a default that doesn't match what we
# start postgres on, and connections are refused.
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

############################## PER-NODE HELPERS
#
# Every helper that touches a remote node does so via
#     srun --overlap -n1 -N1 -w "$host" bash -lc "..."
# By centralising these here we guarantee the head and every worker get the
# same recipe (DB staging, postgres start, ray bring-up, cleanup), so any
# cluster-specific workaround only needs to be applied once.
#
# Helpers that background a srun (`srun ... &`) capture $! into a caller-
# supplied variable name via bash namerefs (`local -n`); callers can then
# `wait "$pid"` on the srun, since it's still a child of the orchestrator
# shell.
#
# All sruns use --overlap because they share each node's allocation with
# other long-lived steps (postgres, ray head/worker). Without --overlap
# slurm refuses with "step creation temporarily disabled".

# stage_local_db_on_node <host>
#   Sync sandbox.tar.zst to <host>'s scratch root and unpack it under
#   $SCRATCH_ROOT/pgdb/, producing $sandbox_dir as the postgres rootfs.
#   Also pre-creates $training_scratch and $NODE_LOCAL_STORE_ROOT so
#   subsequent --bind mounts (which require the host source to already
#   exist) don't fail. Synchronous: returns when extraction is done.
stage_local_db_on_node() {
    local host=$1

    echo "Copying local database to $host"
    srun --overlap -n1 -N1 -w "$host" bash -lc "
        mkdir -p $SCRATCH_ROOT/pgdb $training_scratch $NODE_LOCAL_STORE_ROOT
        rsync -avz --stats $database_sandbox $sandbox_tar
    "

    echo "Unpacking local database on $host"
    srun --overlap -n1 -N1 -w "$host" bash -lc "
        zstd -dc $sandbox_tar | tar -xf - -C $SCRATCH_ROOT/pgdb
    "
}

# start_local_db_on_node <host> <out_pid_var>
#   Start the postgres apptainer instance on <host> and run postgres in the
#   foreground inside the same srun step (so the slurm step's cgroup keeps
#   the instance + postgres alive for the whole job). Backgrounded;
#   <out_pid_var> receives the srun bg pid.
#
#   Postgres needs --writable because its data dir lives inside the sandbox.
#   The pre-mkdir of /global, /clusterfs, /scratch, /dev/shm + the env -u of
#   APPTAINER_BIND/APPTAINER_BINDPATH/SINGULARITY_BIND/SINGULARITY_BINDPATH
#   mirror the well-tested mitigation already used in
#   cluster/ray_local_cluster.sh and scripts/db/local_start_sandbox.sh —
#   without them apptainer FATALs trying to bind host paths whose
#   destinations the sandbox image is missing.
start_local_db_on_node() {
    local host=$1
    local -n out_pid=$2

    echo "Starting local database on $host"
    srun --overlap -n1 -N1 -w "$host" bash -lc "
        apptainer instance stop mysql >/dev/null 2>&1 || true
        mkdir -p $sandbox_dir/global \
                 $sandbox_dir/clusterfs \
                 $sandbox_dir/scratch \
                 $sandbox_dir/dev/shm
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

# wait_for_local_db_on_node <host> <host_ip>
#   Block until pg_isready (run inside the mysql apptainer instance on
#   <host>) reports the DB is accepting TCP connections at <host_ip>, or
#   fail after ~120s. Single srun --overlap so we don't hammer slurm's
#   step-creation rate limiter.
wait_for_local_db_on_node() {
    local host=$1
    local host_ip=$2

    echo "Waiting for local database on $host to be ready"
    srun --overlap -n1 -N1 -w "$host" bash -lc "
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
#   Bring up the ray head (which also starts prometheus + grafana, see
#   ray_start_cluster.sh). Backgrounded; <out_pid_var> receives the srun
#   bg pid. ray_start_cluster.sh blocks on `ray start --block`, so the
#   slurm step persists for the life of the job.
start_ray_head_on_node() {
    local host=$1
    local -n out_pid=$2

    srun --overlap -n1 -N1 -w "$host" bash -lc "
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
#   Bring up a ray worker on <host>. The worker_id is used to give each
#   worker its own host-side tmpdir ($outdir/ray_worker_<id>) bound onto
#   the shared container-side tmpdir, and to name the cleanup pid file
#   (cleanup_<id>.pid) consumed by cleanup_node().
start_ray_worker_on_node() {
    local host=$1
    local worker_id=$2
    local -n out_pid=$3

    echo "Starting worker on: $host"
    srun --overlap -n1 -N1 -w "$host" bash -lc "
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
#   Single-shot ray status gate: block until all <required_nodes> nodes
#   are Active or until ray_check_status.sh times out.
wait_for_ray_cluster() {
    local required=$1

    srun --overlap -n1 -N1 -w "$head_node" bash -lc "
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

# cleanup_node <host> <tmpdir_bindsrc> <pidfile_basename> <out_pid_array>
#   Tear down ray and the local postgres instance on <host>. The cleanup
#   pid file lives at <tmpdir>/<pidfile_basename> inside the container; on
#   the host side that's <tmpdir_bindsrc>/<pidfile_basename>.
#   Backgrounded; pushes the bg srun pid into <out_pid_array> (a nameref
#   to a bash array in the caller's scope).
cleanup_node() {
    local host=$1
    local tmpdir_bindsrc=$2
    local pidfile=$3
    local -n out_pids=$4

    srun --overlap -n1 -N1 -w "$host" bash -lc "
        apptainer exec --userns --nv \
            --bind $storage_server \
            --bind $workspace \
            --bind $bind \
            --bind $tmpdir_bindsrc:$tmpdir \
            $env bash -lc '
            pf=\"$tmpdir/$pidfile\"
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
    out_pids+=($!)
}

# do_cleanup
#   Top-level cleanup orchestrator: tear down head + every worker in
#   parallel, then scancel the slurm job. workers may be unset/empty if
#   we abort before the worker loop runs — the for loop just becomes a
#   no-op in that case.
do_cleanup() {
    local cleanup_jobs=()

    cleanup_node "$head_node" "$outdir" "cleanup_head.pid" cleanup_jobs

    local i=0
    for host in "${workers[@]:-}"; do
        cleanup_node "$host" "$outdir/ray_worker_$i" "cleanup_${i}.pid" cleanup_jobs
        i=$((i+1))
    done

    for pid in "${cleanup_jobs[@]}"; do
        wait "$pid" || true
    done

    sleep 120

    echo "Shutting down the job"
    scancel "$SLURM_JOB_ID"
}

# run_user_task
#   Apptainer-exec the user task ($tasks) on the head node (where the
#   orchestrator already runs). Bind set mirrors start_ray_head_on_node so
#   file paths inside ray actors and inside the user task line up.
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

############################## START HEAD NODE

stage_local_db_on_node "$head_node"
start_local_db_on_node "$head_node" db_head_bg_pid

# Deliberately do NOT export SUPABASE_LOCAL_HOST.
#
# data/databases/local_database.py constructs its connection URI from
# utils.context.node_ip(), which:
#   1. honours $SUPABASE_LOCAL_HOST as a hard override (any value),
#   2. otherwise asks ray for the calling actor's NodeManagerAddress,
#   3. otherwise falls back to a socket lookup on the local hostname.
#
# We start a postgres apptainer instance on every node (head + workers), so
# the correct behaviour is to let each ray actor resolve its OWN node's IP
# via path (2) and query the postgres running on its own node. Exporting
# SUPABASE_LOCAL_HOST=$head_node_ip would force every actor on every worker
# to query the head, defeating the per-worker DBs and turning the head into
# a connection bottleneck.
#
# Note: SUPABASE_LOCAL_URI is unread by production code (only consumed by
# scripts/db/local_start_sandbox.sh and CI), so we don't bother exporting
# it either.

start_ray_head_on_node "$head_node" head_bg_pid

sleep 60

# Fail fast if the postgres srun already died (e.g. apptainer instance start
# couldn't satisfy a system bind path). Without this we'd spend the full
# pg_isready timeout waiting on a backend that's never coming back.
if ! kill -0 "$db_head_bg_pid" 2>/dev/null; then
    wait "$db_head_bg_pid"
    db_rc=$?
    echo "Local DB step on $head_node exited early with code $db_rc; aborting" >&2
    do_cleanup
    exit "$db_rc"
fi

# Initial ray-up loop: cheap `ray status` check from the orchestrator shell
# (not via srun) until the head responds. This is the lightest possible gate
# before we start spinning up workers; the more thorough multi-node check
# happens via wait_for_ray_cluster below.
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
    # Resolve worker host -> IPv4 once; needed below for the per-worker
    # pg_isready gate (apptainer's pg_isready expects -h <ip>, not a
    # hostname that the container can't resolve).
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

# trap 'do_cleanup' EXIT
trap 'do_cleanup; exit 130' INT  # SIGINT
trap 'do_cleanup; exit 143' TERM # SIGTERM like bkill
trap 'do_cleanup; exit 140' TERM # TERM_RUNLIMIT

# Multi-node ray cluster status gate.
wait_for_ray_cluster "$nodes"
rc=$?
if [ $rc -ne 0 ]; then
    echo "Cluster failed to start correctly, exiting"
    do_cleanup
    exit $rc
fi

# DB readiness gate, fanned out across the head + every worker in parallel.
# We need this on every node now because each ray actor queries the postgres
# on its OWN node (see SUPABASE_LOCAL_HOST design note above). If any node's
# postgres failed to come up, fail fast here with a clear error rather than
# letting the corresponding train worker discover it via "Connection refused".
db_check_pids=()
# workers / worker_ips are always declared as `()` above, so a 0-element
# expansion correctly contributes 0 words (no phantom empty strings) for
# the single-node case.
db_check_hosts=("$head_node" "${workers[@]}")
db_check_ips=("$head_node_ip" "${worker_ips[@]}")

for idx in "${!db_check_hosts[@]}"; do
    host=${db_check_hosts[$idx]}
    host_ip=${db_check_ips[$idx]}
    # Each wait_for_local_db_on_node call internally launches a single
    # `srun --overlap` that loops pg_isready inside the mysql instance.
    # Wrap in a subshell + & so we can fan them out and collect results.
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