export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export NCCL_P2P_LEVEL=NVL

while getopts ":i:p:d:c:g:t:" option;do
    case "${option}" in
    i) i=${OPTARG}
       ip=$i
       echo ip=$ip
    ;;
    p) p=${OPTARG}
       port=$p
       echo port=$port
    ;;
    d) d=${OPTARG}
       dashboard_port=$d
       echo dashboard_port=$dashboard_port
    ;;
    c) c=${OPTARG}
       cpus=$c
       echo cpus=$cpus
    ;;
    g) g=${OPTARG}
       gpus=$g
       echo gpus=$gpus
    ;;
    t) t=${OPTARG}
       tmpdir=$t
       echo tmpdir=$tmpdir
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done

cluster_address="$ip:$port"

echo "Starting ray head node @ $(hostname) => $cluster_address with CPUs[$cpus] & GPUs [$gpus]"
job="ray start --head --node-ip-address=$ip --port=$port --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 --min-worker-port 18999 --max-worker-port 19999 --temp-dir=$tmpdir --num-cpus=$cpus --num-gpus=$gpus"
echo $job
$job &

sleep infinity
