export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while getopts ":e:b:n:c:g:o:w:" option;do
    case "${option}" in
    e) e=${OPTARG}
       env=$e
       echo env=$env
    ;;
    b) b=${OPTARG}
       bind=$b
       echo bind=$bind
    ;;
    n) n=${OPTARG}
       nodes=$n
       echo nodes=$nodes
    ;;
    c) c=${OPTARG}
       cpus=$c
       echo cpus=$cpus
    ;;
    g) g=${OPTARG}
       gpus=$g
       echo gpus=$gpus
    ;;
    o) o=${OPTARG}
        outdir=$o
    ;;
    w) w=${OPTARG}
        workload=$w
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done

tmpdir=/tmp/symlink_$(uuidgen | cut -d "-" -f5)
echo "Create symlink: $outdir -> $tmpdir"

############################## SETUP PORTS

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

head_cpus=$(( cpus + 1 )) # 1 cpu core for the training coordinator
apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ./ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $gpus -t $tmpdir &
sleep 10

############################## RUN METRICS

# apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ray metrics launch-prometheus

############################## ADD WORKER NODES

worker_ids=()
num_workers=$((nodes - 1))
for i in $(seq 1 $num_workers)
do
    mkdir -p "${outdir}/ray_worker_${i}"
    echo "Adding worker: ${outdir}/ray_worker_${i}"
    job="bsub -cwd "$(pwd)" -q $LSB_QUEUE -J "${outdir}/ray_worker_${i}" -n $cpus -gpu "num=$gpus:mode=shared" -o "${outdir}/ray_worker_${i}.log" apptainer exec --userns --nv --bind $bind --bind $outdir/ray_worker_${i}:$tmpdir $env ./ray_start_worker.sh -a $cluster_address -c $cpus -g $gpus -t $tmpdir"
    echo $job
    $job


    jid=$(bjobs -r -J "${outdir}/ray_worker_${i}" | awk 'NR==2 {print $1;}')
    while [ -z "$jid" ]
    do
        sleep 1
        jid=$(bjobs -r -J "${outdir}/ray_worker_${i}" | awk 'NR==2 {print $1;}')
    done

    worker_ids+=($jid)
    echo "Running ray_worker_${i} @ ${jid}"
done

############################## CHECK STATUS

apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ./ray_check_status.sh -a $cluster_address -r $nodes

############################## RUN WORKLOAD

echo "Running user workload"
echo $workload
apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env $workload

############################## CLEANUP

echo "Stop ray"
ps aux | grep prometheus | awk '{print $2}' | xargs kill -9
apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ray stop --force

echo "Shutting down the Job"

for jid in "${worker_ids[@]}"
do
    bkill $jid
done

bkill $LSB_JOBID