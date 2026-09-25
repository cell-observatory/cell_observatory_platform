"""For the heavy ops in a torch.profiler chrome trace, attribute time to the enclosing Python frames.

    python scripts/utils/trace_callers.py <trace.json> [--ops aten::copy_,aten::to] [--min-ms 5]

Prints, per op name: total ms, and the top enclosing python_function frames (innermost first) by
summed op duration, plus the input dims seen. Also lists the longest single events.
"""
from __future__ import annotations
import argparse, bisect, json, sys
from collections import defaultdict

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--ops", default="aten::copy_,aten::to,aten::floor_divide,aten::sub,aten::sin,aten::cumsum,aten::searchsorted,aten::cat"); ap.add_argument("--min-ms", type=float, default=5.0); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)
    ops = set(a.ops.split(","))
    with open(a.trace) as f: ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    py = defaultdict(list)  # tid -> python_function events sorted by ts
    for e in ev:
        if e.get("cat") == "python_function": py[e.get("tid")].append(e)
    for t in py: py[t].sort(key=lambda e: e["ts"])
    starts = {t: [e["ts"] for e in v] for t, v in py.items()}
    def frames(e, n=6):
        v, st = py.get(e.get("tid"), []), starts.get(e.get("tid"), [])
        i = bisect.bisect_right(st, e["ts"]); out = []
        for k in range(i - 1, -1, -1):
            p = v[k]
            if p["ts"] + p["dur"] >= e["ts"] + e["dur"]:
                nm = p["name"]
                if "site-packages/torch" in nm or nm.startswith("<built-in") or "profiler" in nm: continue
                out.append(nm.split("/")[-1][:90])
                if len(out) >= n: break
        return " <- ".join(out) or "(no python frame)"
    tot = defaultdict(float); by_frame = defaultdict(lambda: defaultdict(float)); dims = defaultdict(lambda: defaultdict(float)); longest = []
    for e in ev:
        if e.get("cat") != "cpu_op" or e["name"] not in ops: continue
        d = e["dur"] / 1e3; tot[e["name"]] += d
        if d >= a.min_ms:
            fr = frames(e); by_frame[e["name"]][fr] += d
            dims[e["name"]][json.dumps(e.get("args", {}).get("Input Dims"))[:100]] += d
            longest.append((d, e["name"], fr))
    for name in sorted(tot, key=lambda k: -tot[k]):
        print(f"\n### {name}: {tot[name]:.0f} ms total (events >= {a.min_ms} ms attributed below)")
        for fr, d in sorted(by_frame[name].items(), key=lambda kv: -kv[1])[: a.top]: print(f"  {d:8.0f} ms  {fr}")
        for dm, d in sorted(dims[name].items(), key=lambda kv: -kv[1])[:6]: print(f"     dims {d:8.0f} ms  {dm}")
    print("\n### longest single events"); 
    for d, name, fr in sorted(longest, reverse=True)[:15]: print(f"  {d:8.0f} ms  {name}  {fr}")
if __name__ == "__main__": sys.exit(main())
