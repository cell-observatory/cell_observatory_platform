"""Turn SAM2 stage-0 bench runs into matrix rows.

    python scripts/utils/bench_table.py <root> [<root> ...] [--min-iter 20]

Walks <root> for `logs/scalars/step_logbook.csv`, takes medians over iter >= --min-iter, and
reads the setup (backend, shape, bs, mm, levers) from the run config: the yaml manager.py saves
next to a multi_run point, else the stage-0a leaf with the run directory's name. Prints markdown.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
LEAF_DIRS = [f"experiments/janelia/tests/2026_09_02/sam2/{d}" for d in ("diag", "stage0c", "stage0a")] + [
    "experiments/janelia/tests/2026_09_03/sweep", "experiments/janelia/tests/2026_09_03/sweep/pretrain"
]
HBM_GIB = 275.0


def _load_cfg(run_dir: Path):
    yamls = sorted(run_dir.glob("*.yaml"))
    if yamls:
        return OmegaConf.load(yamls[0])
    from hydra import compose, initialize_config_dir

    name = run_dir.name  # leaf name, or the experiment_name of a base run
    leaf = {"bench_local": "local_base", "bench_local_ds": "local_ds_base"}.get(name, name)
    if leaf.startswith("bench_local_"):
        leaf = "shapes/" + leaf[len("bench_local_"):]
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        for d in LEAF_DIRS:
            if (REPO / "configs" / d / f"{leaf}.yaml").exists():
                return compose(config_name=f"{d}/{leaf}.yaml")
    raise FileNotFoundError(f"no leaf yaml for run dir {run_dir.name} under {LEAF_DIRS}")


def _setup(cfg) -> dict:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("now", lambda f: f, replace=True)
    sel = lambda k, d=None: OmegaConf.select(cfg, k, default=d)  # noqa: E731
    gpus = int(sel("clusters.gpus_per_worker", 8)) * int(sel("clusters.worker_nodes", 1))
    bs_total = int(sel("clusters.batch_size", 0))
    shape = list(sel("datasets.input_shape", [0, 0, 0, 0, 0]))[1:4]
    mode = sel("datasets.databases.sample_type", "cube")
    tl = [t.get("_target_", "").split(".")[-1] for t in (sel("datasets.preprocessor.transforms_list") or [])]
    if mode == "tile":
        mode = "tile-crop" if "Crop" in tl else ("tile-resize" if "Resize" in tl else "tile")
    return {
        "backend": "DS" if str(sel("backend", "")) == "DEEPSPEED" else "torch",
        "shape": f"{mode} {shape[0]}x{shape[1]}x{shape[2]}",
        "vox": shape[0] * shape[1] * shape[2],
        "bs": bs_total // max(gpus, 1),
        "mm": sel("datasets.preprocessor.max_masks"),
        "compile": bool(sel("optimizations.models.sam.torch_compile.enable", False)),
        "ac": bool(sel("optimizations.models.sam.activation_checkpoint.enable", False)),
        "acdec": bool(sel("models.meta_arch.sam.use_act_ckpt_iterative_pt_sampling", False)),
        "fp8": bool(sel("parallelism.quantize.enable", False)),
    }


def _row(run_dir: Path, min_iter: int) -> dict:
    d = pd.read_csv(run_dir / "logs" / "scalars" / "step_logbook.csv")
    s = d[d["iter"] >= min_iter]
    m = lambda c, d_=math.nan: float(s[c].median()) if c in s and len(s) else d_  # noqa: E731
    st = m("step_timing/step_time_sec_median")
    dt = m("step_timing/data_time_sec_median")  # spans the previous run_step + the fetch
    r = {"run": run_dir.name, "steps": int(d["iter"].max()) + 1, "steady": len(s)}
    # drift check: median step time of the first min_iter steps vs the rest
    early = d[d["iter"] < min_iter]["step_timing/step_time_sec_median"].median()
    r["early/steady step ms"] = f"{1000 * early:.0f} / {1000 * st:.0f}" if len(s) and not math.isnan(early) else ""
    r.update(_setup(_load_cfg(run_dir)))
    res = m("step/memory/max_reserved(GiB)_median", m("step_system/max_reserved_mem_GB_median"))
    r.update(
        {
            "max_reserved GiB (%)": f"{res:.1f} ({100 * res / HBM_GIB:.0f} %)",
            "step ms": round(1000 * st),
            "fwd / bwd ms": f"{m('step/perf/fwd_time(ms)_median'):.0f} / {m('step/perf/bwd_time(ms)_median'):.0f}",
            "wait ms": round(1000 * (dt - st)) if not math.isnan(dt) else "",
            "samples/s/GPU": round(r["bs"] / st, 2) if st else "",
            "Mvox/s/GPU": round(r["bs"] * r["vox"] / st / 1e6, 1) if st else "",
            "retries / OOM": f"{int(s['step/memory/num_alloc_retries_median'].max())} / {int(s['step/memory/num_ooms_median'].max())}"
            if "step/memory/num_ooms_median" in s and len(s)
            else "",
        }
    )
    r.pop("vox")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--min-iter", type=int, default=50)
    a = ap.parse_args(argv)
    rows = []
    for root in a.roots:
        for csv in sorted(Path(root).rglob("logs/scalars/step_logbook.csv")):
            run_dir = csv.parents[2]
            try:
                rows.append(_row(run_dir, a.min_iter))
            except Exception as e:  # keep going: one broken run must not hide the others
                rows.append({"run": run_dir.name, "steps": f"ERR {type(e).__name__}: {e}"})
    if not rows:
        print("no step_logbook.csv under", a.roots, file=sys.stderr)
        return 1
    df = pd.DataFrame(rows).fillna("")
    cols = list(df.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
