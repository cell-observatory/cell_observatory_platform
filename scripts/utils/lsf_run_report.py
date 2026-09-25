"""Health check + plain-English progress report for chained LSF training runs.

    python scripts/utils/lsf_run_report.py --runs <outdir> [<outdir> ...] --mode report|check [--slack] [--state FILE]

report: one paragraph per run (job state, steps, step time, memory, throughput, train loss trend, val loss, checkpoints).
check : prints only problems (no running job and not done, chain stopped, OOM/error lines in the run log, no new step for
        --stall-min minutes while a job runs, epoch with rising val loss); exit code 1 if any. With --state the check
        remembers what it already reported so each problem is sent once.
--slack pipes the text to ~/.local/bin/claude-slack. Light enough for the login node (bjobs + small csv reads).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def bjobs(name: str) -> list[tuple[str, str, str]]:
    try:
        out = subprocess.run(["bjobs", "-w", "-J", name], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) >= 6:
            rows.append((f[0], f[2], f[5]))
    return rows


def logbook(run: Path, name: str) -> pd.DataFrame | None:
    p = run / "logs" / "scalars" / f"{name}.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def col(d: pd.DataFrame, *keys: str) -> str | None:
    for c in d.columns:
        if all(k in c for k in keys):
            return c
    return None


def loader_progress(run: Path) -> tuple[int, int, float] | None:
    """(rows consumed per rank, rows per rank in the epoch, minutes since the log last moved) from the newest W&B
    output.log tail. Metrics are only flushed at epoch end (PeriodicWriter.after_epoch), so this Ray Data progress
    counter is the live signal within an epoch."""
    logs = sorted(run.glob("logs/wandb/run-*/files/output.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    p = logs[-1]
    try:
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - 200_000))
            tail = f.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    x = total = None
    for m in re.finditer(r'"x": (\d+), "pos": \d+, "desc": "- MapBatches\(LoaderActor\)[^"]*", "total": (\d+)', tail):
        x, total = int(m.group(1)), int(m.group(2))
    if x is None:
        return None
    return x, total, (time.time() - p.stat().st_mtime) / 60


def summarize(run: Path, stall_min: float, light: bool = False) -> tuple[str, list[str]]:
    name = run.name
    jobs = bjobs(name)
    running = [j for j in jobs if j[1] == "RUN"]
    pending = [j for j in jobs if j[1] == "PEND"]
    ckpts = sorted(int(p.name.split("-")[-1]) for p in run.glob("checkpoints/step-*") if p.name.split("-")[-1].isdigit())
    done = (run / "TRAINING_DONE").exists()
    stopped = (run / "CHAIN_STOP").exists()
    problems: list[str] = []
    parts = [f"{name}:"]
    if running:
        parts.append("job " + ", ".join(f"{j[0]} {j[1]} on {j[2]}" for j in running))
    elif done:
        parts.append("finished")
    elif pending:
        parts.append("queued (first link pending, not started yet)")
    else:
        parts.append("NO RUNNING JOB")
    if running and pending:
        parts.append(f"{len(pending)} follower pending")
    elif running and not done:
        problems.append(f"{name}: running link has no pending follower (chain will not continue)")
    if stopped:
        problems.append(f"{name}: CHAIN_STOP marker present")
    if not running and not pending and not done:
        problems.append(f"{name}: no running job and TRAINING_DONE absent")

    d = None if light else logbook(run, "step_logbook")
    if d is not None and len(d):
        it = col(d, "iter") or d.columns[0]
        st = col(d, "step_time_sec_median")
        mem = col(d, "max_reserved(GiB)_median")
        loss = col(d, "step_loss/step_loss")
        n = len(d)
        last = d.tail(100)
        bs = 64
        if st:
            s = float(last[st].median())
            parts.append(f"{n} steps, {s:.2f} s/step (~{bs / s:.0f} samples/s)")
        if mem:
            parts.append(f"{float(last[mem].median()):.0f} GiB")
        if loss:
            first = float(d[loss].head(20).median())
            recent = float(last[loss].median())
            parts.append(f"train loss {first:.2f} -> {recent:.2f} (first 20 vs last 100 steps)")
            for k, lab in (("loss_mask", "mask"), ("loss_dice", "dice"), ("loss_iou", "iou")):
                c = col(d, k)
                if c:
                    parts.append(f"{lab} {float(last[c].median()):.3f}")
            if recent != recent or recent > 1e4:
                problems.append(f"{name}: train loss is nan/exploding ({recent})")
    else:
        parts.append("no step metrics yet (written at epoch end)")
    lp = loader_progress(run)
    if lp:
        x, total, age = lp
        parts.append(f"epoch progress {x}/{total} rows per rank ({100 * x / max(total, 1):.0f} %)")
        if running and age > stall_min:
            problems.append(f"{name}: loader progress log silent for {age:.0f} min while job runs")

    e = None if light else logbook(run, "epoch_logbook")
    if e is not None and len(e):
        vc = col(e, "val") or col(e, "loss")
        if vc:
            vals = [f"{v:.3f}" for v in e[vc].tail(5)]
            parts.append(f"val loss per epoch {vals}")
            if len(e) >= 3 and e[vc].iloc[-1] > e[vc].iloc[-2] > e[vc].iloc[-3]:
                problems.append(f"{name}: val loss rising for 3 epochs")
    parts.append(f"checkpoints {len(ckpts)}" + (f" (last step {ckpts[-1]})" if ckpts else ""))
    for log in run.glob("*.log"):
        if log.name.startswith(("grafana", "prometheus")):
            continue
        # the driver log is hundreds of MB of Ray progress bars: only the tail is read (failures land at the end)
        try:
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 1_000_000))
                txt = f.read()
        except Exception:  # noqa: BLE001
            continue
        oom = txt.count(b"OutOfMemoryError")
        snip = txt.count(b"Error Snippet")
        if oom or snip:
            problems.append(f"{name}: {oom} OOM / {snip} error-snippet lines in the tail of {log.name}")
    return "; ".join(parts), problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--mode", choices=("report", "check"), default="report")
    ap.add_argument("--slack", action="store_true")
    ap.add_argument("--stall-min", type=float, default=45.0)
    ap.add_argument("--state", type=Path, default=None, help="json file remembering problems already sent (check mode)")
    ap.add_argument("--title", default="LR sweep")
    a = ap.parse_args(argv)

    lines, problems = [], []
    for r in a.runs:
        s, p = summarize(r, a.stall_min, light=(a.mode == "check"))
        lines.append(s)
        problems.extend(p)

    if a.mode == "check":
        seen = set()
        if a.state and a.state.exists():
            seen = set(json.loads(a.state.read_text()))
        new = [p for p in problems if p not in seen]
        if a.state:
            a.state.write_text(json.dumps(sorted(seen | set(problems))))
        if not new:
            return 0
        text = f"{a.title} PROBLEM ({time.strftime('%H:%M')}): " + " | ".join(new)
    else:
        text = f"{a.title} hourly report ({time.strftime('%Y-%m-%d %H:%M')}). " + " || ".join(lines)
        if problems:
            text += " || Problems: " + " | ".join(problems)
    print(text)
    if a.slack:
        subprocess.run([str(Path.home() / ".local/bin/claude-slack"), text], check=False, timeout=60)
    return 1 if (a.mode == "check" and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
