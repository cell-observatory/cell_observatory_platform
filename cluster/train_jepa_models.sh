#!/bin/bash

SHAPE=64
MODES=15
DEFAULT='--amp fp16 --opt lamb --lr 5e-4 --wd 5e-5 --jepa'
CLUSTER='lsf'
SUBSET='variable_object_size_fourier_filter_125nm_dataset'
NODES=1
PARTITION='gpu_h100'
BATCH=16384

if [ $CLUSTER = 'slurm' ];then
  HOMEDIR="/clusterfs/nvme/thayer"
  REPO="$HOMEDIR/platform"
  OUTDIR="$HOMEDIR/pretrained_models"
  DATASET="$HOMEDIR/dataset"
  RAY_TEMPLATE="--ray $REPO/cluster/ray_slurm_cluster.sh"
  REQS="--partition abc_a100 --mem=500GB --cpus 16 --gpus 4 --nodes $NODES"
  APPTAINER="--apptainer $HOMEDIR/develop_torch_cuda_12_8.sif"

elif [ $CLUSTER = 'lsf' ];then
  HOMEDIR="/groups/betzig/betziglab/thayer"
  REPO="$HOMEDIR/platform"
  OUTDIR="$HOMEDIR/pretrained_models"
  DATASET="$HOMEDIR/dataset"
  RAY_TEMPLATE="--ray $REPO/cluster/ray_lsf_cluster.sh"
  APPTAINER="--apptainer $HOMEDIR/develop_torch_cuda_12_8.sif"

  if [ $PARTITION = 'gpu_h100_parallel' ];then
    REQS="--partition $PARTITION --nodes $NODES"
  elif [ $PARTITION = 'gpu_a100_parallel' ];then
    REQS="--partition $PARTITION --nodes $NODES"
  elif [ $PARTITION = 'gpu_h100' ];then
    REQS="--partition $PARTITION --gpus 8 --cpus 32 --nodes $NODES"
  else
    REQS="--partition $PARTITION --gpus 4 --cpus 16 --nodes $NODES"
  fi

else
  DATASET="../dataset/training"
fi

DATA="$DATASET/$SUBSET/train/YuMB_lambda510/z200-y125-x125/z$MODES"
CONFIG=" --modes ${MODES} --dataset ${DATA} --input_shape ${SHAPE} --outdir ${OUTDIR} "

if [ $PARTITION = 'gpu_h100' ];then
    AVAL=$(bhosts -o "host_name run:-6"  h100s | grep -w "0" | awk '{print $1}' | wc -l)
    while [ $AVAL -lt $NODES ]
    do
        sleep 1
        echo "Waiting for [$AVAL/$NODES] nodes to be available"
        AVAL=$(bhosts -o "host_name run:-6"  h100s | grep -w "0" | awk '{print $1}' | wc -l)
    done
elif [ $PARTITION = 'gpu_a100' ];then
    AVAL=$(bhosts -o "host_name run:-6"  a100s | grep -w "0" | awk '{print $1}' | wc -l)
    while [ $AVAL -lt $NODES ]
    do
        sleep 1
        echo "Waiting for [$AVAL/$NODES] nodes to be available"
        AVAL=$(bhosts -o "host_name run:-6"  a100s | grep -w "0" | awk '{print $1}' | wc -l)
    done
fi

for NETWORK in 'jepa-base' #'jepa-tiny' 'jepa-small' 'jepa-large'
do
  for PATCH in 32
  do
    python manager.py $CLUSTER $APPTAINER $RAY_TEMPLATE train.py $REQS \
    --task "--network ${NETWORK} --patches ${PATCH} $CONFIG $DEFAULT --batch_size $BATCH" \
    --taskname $NETWORK \
    --name new/$SUBSET/$NETWORK-$PATCH-$MODES
  done
done
