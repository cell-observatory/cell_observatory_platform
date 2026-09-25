"""GPU idle gaps inside a profiled step and what the CPU was doing during them.

    python scripts/utils/trace_gaps.py <trace.json> --step 5 --min-gap-ms 100
"""
from __future__ import annotations
import argparse, bisect, json, sys
from collections import defaultdict

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--step", type=int, default=5); ap.add_argument("--min-gap-ms", type=float, default=100)
    a = ap.parse_args(argv)
    with open(a.trace) as f: ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    steps = [e for e in ev if e.get("name") == f"ProfilerStep#{a.step}" and e.get("cat") == "cpu_op"] or [e for e in ev if e.get("name") == f"ProfilerStep#{a.step}"]
    st = steps[0]; t0, t1 = st["ts"], st["ts"] + st["dur"]
    print(f"step {a.step}: {st['dur']/1e3:.0f} ms wall")
    kern = sorted([e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset") and t0 <= e["ts"] <= t1], key=lambda e: e["ts"])
    comp = [e for e in kern if "nccl" not in e["name"]]
    nccl = [e for e in kern if "nccl" in e["name"]]
    def busy(evs):
        tot, end = 0.0, -1
        for e in evs:
            s, f = e["ts"], e["ts"] + e["dur"]
            if f <= end: continue
            tot += f - max(s, end); end = f
        return tot
    print(f"compute kernels busy {busy(comp)/1e3:.0f} ms, nccl kernels busy {busy(nccl)/1e3:.0f} ms (union), step {st['dur']/1e3:.0f} ms")
    pyf = sorted([e for e in ev if e.get("cat") == "python_function" and t0 <= e["ts"] <= t1], key=lambda e: e["ts"])
    cpu = sorted([e for e in ev if e.get("cat") == "cpu_op" and t0 <= e["ts"] <= t1 and e["dur"] > 20000], key=lambda e: e["ts"])
    def frames_at(t):
        out = []
        for p in pyf:
            if p["ts"] <= t <= p["ts"] + p["dur"]:
                nm = p["name"]
                if "site-packages/torch" in nm or nm.startswith("<built-in") or "profiler" in nm or "module.py" in nm: continue
                out.append(nm.split("/")[-1][:70])
        return " <- ".join(out[-5:][::-1]) or "(none)"
    gaps = []; end = comp[0]["ts"] if comp else t0
    for e in comp:
        if e["ts"] - end >= a.min_gap_ms * 1000: gaps.append((end, e["ts"]))
        end = max(end, e["ts"] + e["dur"])
    print(f"\n## compute-idle gaps >= {a.min_gap_ms} ms (what ran meanwhile)")
    for g0, g1 in gaps:
        mid = (g0 + g1) / 2
        nc = [e["name"].split("(")[0] for e in nccl if e["ts"] < g1 and e["ts"] + e["dur"] > g0]
        ops = [f"{e['name']}({e['dur']/1e3:.0f}ms)" for e in cpu if e["ts"] < g1 and e["ts"] + e["dur"] > g0][:4]
        print(f"  +{(g0-t0)/1e3:7.0f} ms  gap {(g1-g0)/1e3:6.0f} ms  nccl={sorted(set(nc))}  cpu_ops={ops}\n            py: {frames_at(mid)}")
    return 0
if __name__ == "__main__": sys.exit(main())
