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

hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
  echo "Adding host: $host"
  hosts+=($host)
done
echo "The host list is: ${hosts[@]}"
nodes="${#hosts[@]}"

############################## START HEAD NODE

head_node=${hosts[0]}
head_node_ip=$(getent hosts $head_node | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

job="blaunch -z $head_node apptainer exec --userns --nv --bind $bind $env ./ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $cpus -g $gpus &"
echo $job
$job
sleep 10

############################## ADD WORKER NODES

workers=("${hosts[@]:1}")
for host in "${workers[@]}"
do
    echo "Running ray_worker @ ${host}"
    job="blaunch -z $host apptainer exec --userns --nv --bind $bind $env ./ray_start_worker.sh -a $cluster_address -c $cpus -g $gpus &"
    echo $job
    $job
done

############################## CHECK STATUS

apptainer exec --userns --nv --bind $bind $env ./ray_check_status.sh -a $cluster_address -r $nodes

############################## RUN WORKLOAD

echo "Running user workload"
echo $workload
apptainer exec --userns --nv --bind $bind $env $workload

############################## CLEANUP

apptainer exec --userns --nv --bind $bind $env ray stop --force

if [ $? != 0 ]; then
    echo "Failure: $?"
    exit $?
else
    echo "Shutting down the Job"
    bkill $LSB_JOBID
fi
