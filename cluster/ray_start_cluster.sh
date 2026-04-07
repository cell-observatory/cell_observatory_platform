#!/usr/bin/env bash
set -euo pipefail

# NCCL settings optimized for Ethernet without InfiniBand
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while getopts ":i:p:d:c:g:t:q:" option;do
    case "${option}" in
    i)  i=${OPTARG}
        ip=$i
        echo ip=$ip
    ;;
    p)  p=${OPTARG}
        port=$p
        echo port=$port
    ;;
    d)  d=${OPTARG}
        dashboard_port=$d
        echo dashboard_port=$dashboard_port
    ;;
    c)  c=${OPTARG}
        cpus=$c
        echo cpus=$cpus
    ;;
    g)  g=${OPTARG}
        gpus=$g
        echo gpus=$gpus
    ;;
    t)  t=${OPTARG}
        tmpdir=$t
        echo tmpdir=$tmpdir
    ;;
    q)  q=${OPTARG}
        object_store_memory=$(printf "%.0f" "$q")
        echo object_store_memory=$object_store_memory
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

_cleaned=0
cleanup() {
    (( _cleaned )) && return
    _cleaned=1
    echo "Running head node cleanup..."
    ray stop --force >/dev/null 2>&1 || true
    echo "Successfully stopped ray head node"
    python3 /workspace/cell_observatory_platform/utils/cleanup.py
    echo "Successfully ran cleanup.py"
    [ -f "$tmpdir/prometheus.pid" ] && kill "$(cat "$tmpdir/prometheus.pid")" 2>/dev/null || true
    [ -f "$tmpdir/grafana.pid"    ] && kill "$(cat "$tmpdir/grafana.pid")"    2>/dev/null || true
    echo "Successfully stopped prometheus and grafana"
}
trap 'cleanup' EXIT
trap 'cleanup; exit 143' TERM INT

mkdir -p /tmp/ray
cluster_address="$ip:$port"

: "${SUPABASE_LOCAL_PORT:?SUPABASE_LOCAL_PORT must be set in the environment}"
: "${NODE_LOCAL_STORE_ROOT:?NODE_LOCAL_STORE_ROOT must be set in the environment}"

mkdir -p "$NODE_LOCAL_STORE_ROOT"

pick_agent_port() {
    local p=$((dashboard_port + 1))
    # if port lands inside the worker range, bump it out
    if (( p >= 18999 && p <= 19999 )); then
        p=$((20000 + (RANDOM % 10000)))  # 20000–29999
    fi
    echo "$p"
}
DASHBOARD_AGENT_PORT=$(pick_agent_port)

############################## START RAY

# remove any leftover shared memory segments
python3 /workspace/cell_observatory_platform/utils/cleanup.py

echo "Starting ray head node @ $(hostname) => $cluster_address with CPUs[$cpus] & GPUs [$gpus]"
job="ray start --block --head --node-ip-address=$ip --port=$port --dashboard-agent-listen-port=$DASHBOARD_AGENT_PORT --dashboard-host=0.0.0.0 --min-worker-port 18999 --max-worker-port 19999 --temp-dir=$tmpdir --num-cpus=$cpus --num-gpus=$gpus --object-store-memory=$object_store_memory"
echo $job
$job &
head_pid=$!

echo "$$" > "$tmpdir/cleanup_head.pid"

echo "[HEAD NODE]: PID for cleanup is $$"

# wait for the head node to start/create a new session
# directory to ensure that prometheus and grafana 
# are started with the correct session directory
sleep 60

############################## START PROMETHEUS

echo "Starting prometheus on $(hostname) => $cluster_address with dashboard_port[$dashboard_port] & tmpdir[$tmpdir]"

mkdir -p "$tmpdir/prometheus"

# function to get latest session directory
latest_session_dir() {
    root=$1
    [ -n "$root" ] || { echo "usage: latest_session_dir TMPDIR" >&2; return 1; }
    set -- "$root"/session_2*
    [ -e "$1" ] || return 1

    {
        for p do
            b=$(basename "$p")
            key=$(printf %s "$b" |
                    sed -n 's/^session_\([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\).*/\1/p') || key=
            [ -n "$key" ] && printf '%s\t%s\n' "$key" "$p"
      done
    } | LC_ALL=C sort | tail -n1 | awk -F '	' '{print $2}'
}

# get session directory, set prometheus config, and 
# grafana provisioning paths
session_directory="$(latest_session_dir "$tmpdir")"
session_directory="$(readlink -f "$session_directory")"
prometheus_config="$session_directory/metrics/prometheus/prometheus.yml"
grafana_config="$session_directory/metrics/grafana/grafana.ini"
grafana_provisioning_config="$session_directory/metrics/grafana/provisioning"

if [ -z "$prometheus_config" ]; then
    echo "WARN: no Prometheus config found under $tmpdir" >&2
else
    echo "Using Prometheus config: $prometheus_config"

    prometheus --config.file="$prometheus_config" \
        --storage.tsdb.path="$tmpdir/prometheus" \
        --web.enable-lifecycle >"$tmpdir/prometheus.log" 2>&1 &
    echo "$!" > "$tmpdir/prometheus.pid"
fi

############################## START GRAFANA

if [ -z "$grafana_config" ]; then
    echo "WARN: no grafana.ini found under $tmpdir" >&2
else
    echo "Using Grafana config: $grafana_config"
    
    GRAFANA_DATA="$tmpdir/grafana/data"
    GRAFANA_LOGS="$tmpdir/grafana/logs"
    GRAFANA_PLUGINS="$tmpdir/grafana/plugins"
    mkdir -p "$GRAFANA_DATA" "$GRAFANA_LOGS" "$GRAFANA_PLUGINS"

    grafana_homepath=/usr/share/grafana
    export GF_PATHS_DATA="$GRAFANA_DATA"
    export GF_PATHS_LOGS="$GRAFANA_LOGS"
    export GF_PATHS_PLUGINS="$GRAFANA_PLUGINS"
    export GF_PATHS_PROVISIONING="$grafana_provisioning_config"
    /usr/sbin/grafana-server --homepath "$grafana_homepath" --config "$grafana_config" web >"$tmpdir/grafana.log" 2>&1 &
    echo "$!" > "$tmpdir/grafana.pid"
fi

wait "$head_pid"