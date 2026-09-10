#!/bin/bash
# Submit one b300 node job that runs eval leaves one after another in local mode (scripts/utils/run_queue.sh).
# Usage: submit_eval_queue.sh <repo_dir> <cfg_dir relative to configs/> <hours> <leaf> [<leaf> ...]
# Logs: $DATA_DIR/sam2_study/eval/logs/<leaf>_rN.log (one per leaf) + the LSF log next to them.
REPO=$1; CFGD=$2; HOURS=$3; shift 3
LOGD=/groups/betzig/betziglab/hph/cell_observatory_project/sam2_study/eval/logs
mkdir -p "$LOGD"
# same node request as manager.py's LSF launcher (one full b300 node)
bsub -J eval_queue -q gpu_b300 -n 96 -R "span[ptile=96]" -app parallel-96 -gpu "num=8:mode=exclusive_process" \
  -W "${HOURS}:00" -o "$LOGD/lsf_eval_queue_%J.log" \
  "cd $REPO && bash scripts/utils/run_queue.sh $LOGD $CFGD $*; \
   apptainer instance stop -a >/dev/null 2>&1; pkill -u \$USER -f 'postgres|raylet|gcs_server|ray::|dashboard' >/dev/null 2>&1; true"
# the trailing cleanup matters: a leftover postgres / Ray process from the local launcher kept two finished queue jobs
# alive (holding their nodes) on 2026-09-10; the node is exclusive to the job, so killing our own leftovers is safe
