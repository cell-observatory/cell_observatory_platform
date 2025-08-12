# /bin/bash
#SBATCH --job-name=collectives_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8         
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --account=co_abc  
#SBATCG --qos=abc_high
#SBATCH --partition=dgx
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# this script is copied from:
# https://github.com/stas00/ml-engineering
# we use it to run the all-reduce benchmark across 
# multiple nodes using PyTorch's distributed framework

OUTDIR="/clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/networking_benchmarks"

GPUS_PER_NODE=8
NNODES=1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -u -m torch.distributed.run \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_backend c10d \
    --max_restarts 0 \
    "$SCRIPT_DIR/benchmark_collectives.py" --outdir "$OUTDIR"