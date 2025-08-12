# /bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/benchmarks_io/benchmark_io.sh

# adjust below to your needs

# knobs you can tweak
DIR="/clusterfs/vast/Data/fio_tests"
NUM_SAMPLES=32          # how many "zarr samples" you load at once
FILES_PER_SAMPLE=128    # files per sample
CHUNK_SIZE=1M           # 1 MiB "zarr chunk" files

# Total = 32 * 128 * 1MiB = 4096 MiB = about 4 GiB

mkdir -p "$DIR"

# 1) Create the synthetic file set (writes NUM_SAMPLES*FILES_PER_SAMPLE files)
# fio --name=mk \
#     --directory="$DIR" \
#     --ioengine=libaio \
#     --iodepth=64 \
#     --rw=write --bs="$CHUNK_SIZE" \
#     --numjobs="$NUM_SAMPLES" \
#     --nrfiles="$FILES_PER_SAMPLE" \
#     --filesize="$CHUNK_SIZE" \
#     --filename_format=job\$jobnum-file\$filenum \
#     --group_reporting=1

# 2A) Read test (BUFFERED)
# NOTE: (a) file_service_type tells fio to pick which file to issue
#       the next I/O at random rather than walking files in order
#.      (b) openfiles=32 means each job can open 32 files at once
#        why set to 32? rationale: typical Zarr opens a chunk file, reads,
#        then closes. generally doesn’t hold thousands of chunk files open
#        simultaneously. we try to simulate that.
#       (c) --time_based=1 --runtime=60 --ramp_time=5
#       implies run for a fixed time instead of until a byte/IO count 
#       is met.
#       (d) --direct=0 means not direct I/O
#       (e) --group_reporting=1 means report stats for all jobs together
# fio --name=read_cached \
#     --directory="$DIR" \
#     --ioengine=libaio \
#     --iodepth=128 \
#     --rw=randread --bs="$CHUNK_SIZE" \
#     --numjobs="$NUM_SAMPLES" \
#     --nrfiles="$FILES_PER_SAMPLE" \
#     --filesize="$CHUNK_SIZE" \
#     --filename_format=job\$jobnum-file\$filenum \
#     --file_service_type=random \
#     --openfiles=32 \
#     --time_based=1 --runtime=30 --ramp_time=5 \
#     --direct=0 --group_reporting=1

# 2B) Read test (DIRECT I/O)
fio --name=read_direct \
    --directory="$DIR" \
    --ioengine=libaio \
    --iodepth=256 \
    --rw=randread --bs="$CHUNK_SIZE" \
    --numjobs="$NUM_SAMPLES" \
    --nrfiles="$FILES_PER_SAMPLE" \
    --filesize="$CHUNK_SIZE" \
    --filename_format=job\$jobnum-file\$filenum \
    --file_service_type=random \
    --openfiles=32 \
    --time_based=1 --runtime=40 --ramp_time=5 \
    --direct=1 --group_reporting=1


# ------------------------------------ ------------------------------------------ ------------------------------------------

FILE="$DIR/bigfile_4g"

# create a real 4 GiB file on disk
# fio --name=mk --filename="$FILE" --rw=write --bs=4M --size=4G \
#     --ioengine=libaio --direct=1 --iodepth=64 --numjobs=1 --group_reporting=1

# test read performance

# sequential read
# fio --name=seq_roof --filename="$FILE" --rw=read --bs=4M \
#     --ioengine=libaio --direct=1 --iodepth=64 \
#     --time_based=1 --runtime=90 --group_reporting=1

# fio --name=seq_roof \
#     --filename="$FILE" \
#     --rw=read \
#     --bs=32k \
#     --size=4G \
#     --ioengine=libaio \
#     --direct=1 \
#     --iodepth=512 \
#     --numjobs=16 \
#     --time_based=1 \
#     --runtime=30 \
#     --group_reporting=1