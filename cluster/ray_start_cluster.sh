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

mkdir -p /tmp/ray
cluster_address="$ip:$port"

echo "Starting ray head node @ $(hostname) => $cluster_address with CPUs[$cpus] & GPUs [$gpus]"
job="ray start --head --node-ip-address=$ip --port=$port --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 --min-worker-port 18999 --max-worker-port 19999 --temp-dir=$tmpdir --num-cpus=$cpus --num-gpus=$gpus --object-store-memory=$object_store_memory"
echo $job
$job &

echo "Starting prometheus server on $(hostname) => $cluster_address with dashboard_port[$dashboard_port] & tmpdir[$tmpdir]"
prometheus --config.file=$tmpdir/session_latest/metrics/prometheus/prometheus.yml

sleep infinity