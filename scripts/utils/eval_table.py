#!/usr/bin/env python
"""One markdown row per evaluation run (job_type test): the headline instance metrics from
<run>/logs/scalars/epoch_logbook.csv and the step time from step_logbook.csv.

    python scripts/utils/eval_table.py $DATA_DIR/sam2_study/eval            # every run dir below
    python scripts/utils/eval_table.py <run_dir> [<run_dir> ...]
"""
import csv
import statistics
import sys
from pathlib import Path

COLS = [
    ("mask_map", "mask mAP"), ("mask_miou", "mask mIoU"), ("mask_match_recall", "match recall"),
    ("box_map", "box mAP"), ("box_f1", "box F1"),
    ("pred_iou_eval/iou_head_n", "preds"), ("pred_iou_eval/iou_head_mae", "IoU-head MAE"),
    ("pred_iou_eval/iou_head_spearman", "IoU-head Spearman"),
    ("pred_iou_eval/precision@0.5", "precision@0.5"), ("pred_iou_eval/precision@0.75", "precision@0.75"),
]


def run_dirs(args):
    for a in args:
        p = Path(a)
        if (p / "logs" / "scalars" / "epoch_logbook.csv").exists():
            yield p
        else:
            yield from sorted(d for d in p.glob("*") if (d / "logs" / "scalars" / "epoch_logbook.csv").exists())


def row(run: Path):
    ep = list(csv.DictReader(open(run / "logs" / "scalars" / "epoch_logbook.csv")))
    if not ep:
        return None
    r = ep[-1]
    vals = []
    for key, _ in COLS:
        v = r.get(f"epoch_loss/test/{key}_median", "")
        try:
            v = f"{float(v):.0f}" if key.endswith("_n") else f"{float(v):.3f}"
        except ValueError:
            pass
        vals.append(v)
    st_path = run / "logs" / "scalars" / "step_logbook.csv"
    steps, t = 0, ""
    if st_path.exists():
        st = list(csv.DictReader(open(st_path)))
        ts = [float(x["step_timing/test/step_time_sec_median"]) for x in st
              if x.get("step_timing/test/step_time_sec_median")]
        steps = len(st)
        if ts:
            t = f"{statistics.median(ts):.1f}"
    return [run.name, str(steps), t] + vals


def main():
    hdr = ["run", "steps/rank", "s/cube"] + [c for _, c in COLS]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for run in run_dirs(sys.argv[1:] or ["."]):
        r = row(run)
        if r:
            print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
