while getopts ":e:b:c:g:o:w:" option;do
    case "${option}" in
    e) e=${OPTARG}
       env=$e
       echo env=$env
    ;;
    b) b=${OPTARG}
       bind=$b
       echo bind=$bind
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

############################## FIND NODES/HOSTS

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

############################## START HEAD NODE

apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ./ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $cpus -g $gpus -t $tmpdir &
sleep 10

rpids=$(pgrep -u $USER ray)
echo "Ray head node PID:"
echo $rpids


############################## CHECK STATUS

echo apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ./ray_check_status.sh -a $cluster_address -r 1

############################## RUN WORKLOAD

echo "Running user workload"
echo $workload
apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env $workload

############################## CLEANUP

echo "Stop ray"
apptainer exec --userns --nv --bind $bind --bind $outdir:$tmpdir $env ray stop --force