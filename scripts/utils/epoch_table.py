"""Per-epoch validation table for a set of runs, from their local epoch logbooks.

    python scripts/utils/epoch_table.py --runs <outdir> [<outdir> ...] [--md]

Reads only logs/scalars/epoch_logbook.csv (a few KB per run); safe on a login node. Columns: run, epoch, val total /
dice / iou loss, train step loss (median of the epoch), step time, epoch hours.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows_for(run: Path):
    p = run / "logs" / "scalars" / "epoch_logbook.csv"
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(p.open()):
        try:
            out.append(
                dict(
                    run=run.name,
                    epoch=int(float(r["epoch"])) + 1,
                    val=float(r["epoch_loss/val/step_loss_mean"]),
                    dice=float(r["epoch_loss/val/loss_dice_mean"]),
                    iou=float(r["epoch_loss/val/loss_iou_mean"]),
                    train=float(r["step_loss/step_loss_median"]),
                    step_s=float(r["step_timing/step_time_sec_median"]),
                    epoch_h=float(r["epoch_timing/epoch_time_sec_median"]) / 3600,
                )
            )
        except (KeyError, ValueError):
            continue
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args(argv)
    rows = [r for run in a.runs for r in rows_for(run)]
    if a.md:
        print("| run | epoch | val total | val dice | val iou | train | s/step | epoch h |")
        print("|---|---|---|---|---|---|---|---|")
        fmt = "| {run} | {epoch} | {val:.3f} | {dice:.3f} | {iou:.3f} | {train:.3f} | {step_s:.2f} | {epoch_h:.2f} |"
    else:
        print(f"{'run':32s} ep  val    dice   iou    train  s/step epoch_h")
        fmt = "{run:32s} {epoch:<3d} {val:.3f}  {dice:.3f}  {iou:.3f}  {train:.3f}  {step_s:.2f}   {epoch_h:.2f}"
    for r in rows:
        print(fmt.format(**r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
