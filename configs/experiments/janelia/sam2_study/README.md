# SAM2 study (Janelia b300) — configs + plans in one place

| what | where |
|---|---|
| study plan (stages 0–3) | `plans/plan_sam2_v2.md` |
| Stage 0 result: production recipes | `plans/perf_summary.md`; configs `recipe/recipe_r1.yaml` (1 click, bs 8/GPU) and `recipe/recipe_r3.yaml` (3 clicks, bs 5/GPU); bases `recipe/base_study.yaml`, `recipe/bench_base.yaml`; stage-0 leaves `recipe/stage0c/`, `recipe/stage0a/`, `recipe/diag/` |
| setup sweep (bs / levers / bug-flush per run type) | `plans/setup_sweep.md`; configs `sweep/` (SAM2 variants) and `sweep/pretrain/` (MAE/JEPA) |
| Stage 1: learning rate | `plans/stage1_lr_plan.md`; configs `stage1_lr/{lr_base,lr_sweep,chain_smoke}.yaml` |
| Stage 1.5: tile-scale adaptation at 128×384×1024 | `plans/stage1p5_tile_plan.md` (configs to be written under `stage1p5_tile/`) |

Launch from the frozen run worktree (`run_parent/cell_observatory_platform`, branch `runs/sam2-stage1`):
`cd <worktree> && python manager.py --config-name=experiments/janelia/sam2_study/stage1_lr/lr_sweep.yaml`.
Compose-check before every launch (shape, bs, epochs, lr, transforms, excluded tiles, queue, chain) — the inheritance chain is
bring-up base (`tests/2026_09_01/sam2/base_sam2_torch`) → `recipe/base_study` → `recipe/recipe_r1` → stage leaf.
Bench rows: `python scripts/utils/bench_table.py $DATA_DIR/sam2_study/<dir>`.
