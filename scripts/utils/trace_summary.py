"""Summarize a torch.profiler chrome trace (tensorboard_trace_handler output).

    python scripts/utils/trace_summary.py <trace.pt.trace.json> [--top 25]

Prints: profiler steps and their wall time, GPU kernel time by kernel name, CPU-op time (self)
by op name, the largest CPU ops by inclusive time, and the input shapes seen for the top ops.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def _load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)
    ev = [e for e in _load(a.trace).get("traceEvents", []) if e.get("ph") == "X"]

    steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep#")]
    print("## profiler steps")
    for e in steps:
        print(f"{e['name']}: {e['dur'] / 1e3:.0f} ms wall")

    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    total_gpu = sum(e["dur"] for e in kern)
    by_k = defaultdict(lambda: [0.0, 0])
    for e in kern:
        by_k[e["name"]][0] += e["dur"]
        by_k[e["name"]][1] += 1
    n_steps = max(len(steps), 1)
    print(f"\n## GPU time: {total_gpu / 1e3 / n_steps:.0f} ms per step over {n_steps} steps; top kernels (ms/step, calls/step)")
    for name, (d, n) in sorted(by_k.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"{d / 1e3 / n_steps:9.1f}  {n / n_steps:7.1f}  {name[:110]}")

    ops = [e for e in ev if e.get("cat") == "cpu_op"]
    # self time: inclusive minus children (same thread, nested by time)
    ops.sort(key=lambda e: (e.get("tid"), e["ts"], -e["dur"]))
    incl = defaultdict(lambda: [0.0, 0])
    selft = defaultdict(float)
    shapes = defaultdict(set)
    stack = []
    for e in ops:
        while stack and not (stack[-1].get("tid") == e.get("tid") and stack[-1]["ts"] + stack[-1]["dur"] >= e["ts"] + e["dur"]):
            stack.pop()
        if stack:
            stack[-1]["_child"] = stack[-1].get("_child", 0.0) + e["dur"]
        stack.append(e)
        incl[e["name"]][0] += e["dur"]
        incl[e["name"]][1] += 1
        sh = e.get("args", {}).get("Input Dims")
        if sh:
            shapes[e["name"]].add(json.dumps(sh)[:120])
    for e in ops:
        selft[e["name"]] += e["dur"] - e.get("_child", 0.0)
    print(f"\n## CPU ops by self time (ms/step, calls/step)")
    for name, d in sorted(selft.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"{d / 1e3 / n_steps:9.1f}  {incl[name][1] / n_steps:7.1f}  {name[:100]}")
    print(f"\n## CPU ops by inclusive time (ms/step)")
    for name, (d, n) in sorted(incl.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"{d / 1e3 / n_steps:9.1f}  {n / n_steps:7.1f}  {name[:100]}")
    print("\n## input shapes of the top inclusive ops")
    for name, (d, n) in sorted(incl.items(), key=lambda kv: -kv[1][0])[:12]:
        for s in list(shapes.get(name, []))[:3]:
            print(f"  {name[:60]}: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
