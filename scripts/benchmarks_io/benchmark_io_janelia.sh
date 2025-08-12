#!/usr/bin/bash

# knobs you can tweak
DIR="/groups/betzig/betziglab/CellObservatoryData/fio_test"
NUM_SAMPLES=32          # how many "zarr samples" you load at once
FILES_PER_SAMPLE=128    # files per sample
CHUNK_SIZE=1M           # 1 MiB "zarr chunk" files

# Total = 32 * 128 * 1MiB = 4096 MiB = about 4 GiB

mkdir -p "$DIR"

# 1) Create the synthetic file set (writes NUM_SAMPLES*FILES_PER_SAMPLE files)
echo "Creating synthetic file dataset for testing"
fio --name=mk \
   --directory="$DIR" \
   --ioengine=io_uring \
   --iodepth=64 \
   --rw=write --bs="$CHUNK_SIZE" \
   --numjobs="$NUM_SAMPLES" \
   --nrfiles="$FILES_PER_SAMPLE" \
   --filesize="$CHUNK_SIZE" \
   --filename_format=job\$jobnum-file\$filenum \
   --group_reporting=1

echo "Read test (DIRECT I/O) with a single stream"
fio --name=read_direct \
    --directory="$DIR" \
    --iodepth=256 \
    --rw=randread --bs="$CHUNK_SIZE" \
    --numjobs="$NUM_SAMPLES" \
    --nrfiles="$FILES_PER_SAMPLE" \
    --filesize="$CHUNK_SIZE" \
    --filename_format=job\$jobnum-file\$filenum \
    --file_service_type=random \
    --openfiles=32 \
    --time_based=1 --runtime=40 --ramp_time=5 \
    --direct=1 --group_reporting=1 2>&1 | tee "$DIR/fio_read_direct.log"
PID=$!
wait $PID


echo "Read test (DIRECT I/O) with two streams"
fio --name=read_direct \
    --directory="$DIR" \
    --iodepth=256 \
    --rw=randread --bs="$CHUNK_SIZE" \
    --numjobs="$NUM_SAMPLES" \
    --nrfiles="$FILES_PER_SAMPLE" \
    --filesize="$CHUNK_SIZE" \
    --filename_format=job\$jobnum-file\$filenum \
    --file_service_type=random \
    --openfiles=32 \
    --time_based=1 --runtime=40 --ramp_time=5 \
    --direct=1 --group_reporting=1 2>&1 | tee "$DIR/fio_read_direct_worker1.log" & \
fio --name=read_direct \
    --directory="$DIR" \
    --iodepth=256 \
    --rw=randread --bs="$CHUNK_SIZE" \
    --numjobs="$NUM_SAMPLES" \
    --nrfiles="$FILES_PER_SAMPLE" \
    --filesize="$CHUNK_SIZE" \
    --filename_format=job\$jobnum-file\$filenum \
    --file_service_type=random \
    --openfiles=32 \
    --time_based=1 --runtime=40 --ramp_time=5 \
    --direct=1 --group_reporting=1 2>&1 | tee "$DIR/fio_read_direct_worker2.log"

PID=$!
wait $PID
