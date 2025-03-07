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

ln -sf $outdir /tmp/ray_symlink
echo "Create symlink: ray"

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


############################## FIND NODES/HOSTS

set -x

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

############################## START HEAD NODE

job="srun --nodes=1 --ntasks=1 -w $head_node apptainer exec --userns --nv --bind $bind $env ./ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $cpus -g $gpus &"
echo $job
$job

############################## ADD WORKER NODES

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    worker_job="srun --nodes=1 --ntasks=1 -w $node_i apptainer exec --userns --nv --bind $bind $env ./ray_start_worker.sh -a $cluster_address -c $cpus -g $gpus &"
    echo $worker_job
    $worker_job
done

############################## RUN WORKLOAD

echo "Starting workload: $workload"
apptainer exec --userns --nv --bind $bind $env $workload

############################## CLEANUP

apptainer exec --userns --nv --bind $bind $env ray stop --force

if [ $? != 0 ]; then
    echo "Failure: $?"
    exit $?
else
    echo "Shutting down the Job"
    scancel $SLURM_JOB_ID
fi
