export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

############################## FIND NODES/HOSTS

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

############################## START HEAD NODE

apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $gpus -t $tmpdir -q $object_store_memory &
sleep 20

rpids=$(pgrep -u $USER ray)
echo "Ray head node PID:"
echo $rpids

############################## CHECK STATUS

# add exit trap to ensure cleanup on script exit
# this will ensure that we stop the Ray cluster and cancel worker jobs
# even if the script fails at any point henceforth
cleanup() {
    ec=$? # exit code of the last command that triggered the trap
    echo "running cleanup (exit code: $ec)"

    # stop Ray on the head node
    apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ray stop --force
}
trap cleanup EXIT

echo apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env /workspace/cell_observatory_platform/cluster/ray_check_status.sh -a $cluster_address -r 1

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks