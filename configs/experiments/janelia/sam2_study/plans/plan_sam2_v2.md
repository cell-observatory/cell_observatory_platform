# SAM2 study plan v2 — Stage 0 efficiency → Step 1 LR → Step 2 channels → Step 3 scale

2026-09-02 · status: **PROPOSAL; stage-0a configs written (`configs/experiments/janelia/tests/2026_09_02/sam2/`), nothing run** · concise rewrite of `plan_sam2.md`.
Target: Janelia b300 (8× B300, ~275 GB HBM, 4 h wall). Other GPUs by extrapolation (§1.4).

Fixed for the whole study (not ablated):
- **Loss**: PointRend point-sampling path only (`use_point_sampling: true`, `num_points: 26000`,
  `low_res_multimasks: true`, as in the b300 base). Dense path is too slow.
- **Loss weights**: SAM2 defaults, focal 20 / dice 1 / IoU 1.
- **Schedule, wd, clip**: base values (warmup-cosine, wd = 0.1·lr, max_norm 1).

| stage | question | jobs | output |
|---|---|---|---|
| 0a (local, live) | capacity, levers, gates on one node | ~24 runs, 1 node | `2026-09-02-sam2-tranche0-stage1.md` |
| 0b (LSF) | full efficiency matrix | ~55 × 5–10 min | §1.4 table → production config |
| 1 | LR | 4 (+2 seeds) × 4 h | one LR |
| 2 | channel param, channel dropout, interaction | 13 × 4 h | recipe |
| 3 | epochs, pretrained init, ViT-H | ~8 chained | final model |

Data facts: `cod_db_20260824` has only 1×128³ cubes (3.27 M annotated, median 9 instances, max 46);
tiles 128×512×{2560,2816,3328}, median 226 instances/frame. ABC A100 (09-01): cube 128³ bs 1 mm 8 →
3.2 GiB, 2.3 s/step torch / 3.8 s DeepSpeed; tile-resize 128×384×1024 bs 1 mm 8 → 57.5 GiB, mm 16 OOM.

---

## 1. Stage 0 — efficiency matrix

**Goal.** Decide with numbers: sample mode + shape, bs/GPU, `max_masks`, backend, compile, FP8, and
whether the data path keeps the GPU busy (`data_time / step_time`).

### 1.1 How every probe runs

- Step mode, `max_steps: 60` (100 for compile), no val, no checkpoint.
- `with_perf_metrics: true` (torch only: step/fwd/bwd ms, tps, MFU); `torch_memory_stats` every
  10 steps; `iteration_timer` for `data_time` (both backends).
- `nvidia-smi dmon -s um -d 5 -o T` sidecar per job (§1.5). All numbers = median over steps 20+.
- `max_rows ≥ 60 · bs_per_gpu · 8`, `timepoint_list: [0]`. Tile queries cost ~5 min (sort);
  `max_tile_shape: [128, 512, 3328]` keeps the pinned buffer off the 256×1536×2816 outlier.
- `max_masks` is a **cap** (batch-local padding landed in 291abac). Cubes: mm 46 = real count, fill
  VRAM with bs. Tiles: ladder mm 46 → 100, then bs at mm 100. Never above 100.
- Prereqs: `B1` block paths, `B3` batch-local padding, `Crop` + `max_tile_shape` — all landed.
  `B2` dead `torch.cat` — open (+12 % dead memory per tile row until fixed).

### 1.2 Run matrix (stage 0b; stage 0a is the one-point-per-lever subset)

| launcher | axis | pts | fixed |
|---|---|---|---|
| `E1_backend` | {torch FSDP2, DS ZeRO-2} × bs/GPU {1, 2} × mm {16, 46} | 8 | cube 128³ |
| `E2_mode_shape` | {cube 128³, crop 128³, crop 128×256×256, crop 128×384×512, crop 128×384×1024, resize 128×384×1024} × mm {46, 100} | 11 | bs 1, torch (cube row mm 46 only) |
| `E3_bs_mm_cube128` | bs/GPU {1, 2, 4, 8, 16, 32} × mm {16, 46} | 12 | cube 128³ |
| `E4_bs_mm_crop384x512` | bs/GPU {1, 2, 4} × mm {46, 100} | 6 | crop 128×384×512 |
| `E5_compile` | {off, on} × {cube bs_cube mm 46, crop 384×512 bs_crop mm 100} + DS cube | 5 | 100 steps |
| `E6_fp8` | {bf16, mxfp8_cublas, mxfp8_cublas_rceil} × {cube, crop} + keep-heads at mxfp8 | 8 | torch |
| `E7_fit_levers` | act-ckpt {off, on} × `use_act_ckpt_iterative_pt_sampling` {F, T} | 4 | crop 128×384×1024 bs 1 mm 100; only if E2 shows > 80 % HBM |

Cube probe ≈ 5 min, tile ≈ 10 min. Submit E1, E3, E2 first; they fix bs and shape for E4–E7.

"Bigger cubes" = tile crops at native resolution (same voxels a DB-side cut would give, different
I/O path). tile-resize = today's path (X squashed 3.25×, masks resampled).

### 1.3 Decision rules

- **Backend (E1)**: torch unless DS ≥ 15 % faster at same memory (torch alone has FP8, DCP, resume, perf).
- **Shape (E2/E4)**: largest shape with `max_reserved` < 80 % HBM and `data_time/step_time` < 0.2.
  A tile-crop shape that wins on compute but fails the data gate → build crop-at-load; run on cube until then.
- **bs × mm (E3/E4)**: largest bs with samples/s within 10 % of best and `num_alloc_retries == 0`.
- **Compile (E5)**: on if steady-state step ≥ 8 % faster and one DCP save+resume passes.
- **FP8 (E6)**: Step 3 only, if ≥ 15 % faster **and** a 2-epoch loss curve tracks bf16. Steps 1–2 stay bf16.
- **Fit levers (E7)**: only to make a shape fit.

### 1.4 The table

| shape | mode | backend | bs/GPU | mm | prec | compile | max_reserved GiB (%) | step ms | data ms | samples/s/GPU | Mvox/s/GPU | MFU | util % | retries/OOM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Other GPUs: fit `max_reserved ≈ M₀ + a·N·ZYX` from the two mm points per shape, solve N at
0.8 × {80, 141, 192, 275} GB. Cross-check: ABC A100 57.5 GiB at mm 8, OOM at mm 16 (resize 1024).

### 1.5 Config sketches — Stage 0

Stage 0a configs are written under `configs/experiments/janelia/tests/2026_09_02/sam2/` (bases, `shapes/`, `stage0a/`); the stage-0b launchers below go in `.../sam2/stage0b/` on the same bases. Absolute defaults paths only.

```yaml
# 00_common/base_study.yaml
# @package _global_
defaults:
  - /experiments/janelia/tests/2026_09_01/sam2/base_sam2_torch
  - _self_
experiment_name: sam2_study_base
wandb_project: sam2_study
paths: {outdir: "${paths.data_path}/sam2_study/${experiment_name}"}
parallelism: {quantize: {enable: false}}          # bf16 except E6 / Step 3
models:
  meta_arch:
    sam:
      debug: false
      criterion_args: {use_point_sampling: true, num_points: 26000, low_res_multimasks: true}
datasets:
  split: 0.0                  # Steps 1+ set 0.1
  max_rows: null
  timepoint_list: null
  preprocessor: {max_masks: 46}
  databases: {max_tile_shape: null}
schedulers: {epochs: 1, warmup: 0, cooldown: 0}
optimizers: {lr: 1e-4, wd: "${eval:'${optimizers.lr} * 0.1'}"}
```

```yaml
# 00_common/base_study_ds.yaml -- DeepSpeed ZeRO-2 twin (E1, E5 only); no quantize, no perf metrics
# @package _global_
defaults:
  - /experiments/janelia/tests/2026_09_01/sam2/base_sam2_deepspeed
  - _self_
experiment_name: sam2_study_base_ds
wandb_project: sam2_study
paths: {outdir: "${paths.data_path}/sam2_study/${experiment_name}"}
models:
  meta_arch:
    sam:
      debug: false
      criterion_args: {use_point_sampling: true, num_points: 26000, low_res_multimasks: true}
datasets: {split: 0.0, max_rows: null, timepoint_list: null, preprocessor: {max_masks: 46}, databases: {max_tile_shape: null}}
schedulers: {epochs: 1, warmup: 0, cooldown: 0}
optimizers: {lr: 1e-4, wd: "${eval:'${optimizers.lr} * 0.1'}"}
```

```yaml
# 00_bench/bench_base.yaml -- 60-step probe
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/00_common/base_study
  - _self_
experiment_name: sam2_bench
wandb_project: sam2_study_bench
datasets:
  max_rows: 16384             # >= 60 x 32/GPU x 8
  timepoint_list: [0]
trainer_loop:
  iteration_mode: step
  max_steps: 60
  checkpoint_every_n_epochs: null
  checkpoint_save_period: 1000000
  with_perf_metrics: true
hooks:
  hooks_list:
  - name: free_device_buffer
  - name: lr_scheduler
    backend: TORCHTITAN
  - name: iteration_timer
  - name: periodic_writer
  - name: torch_memory_stats
    step_period: 10           # must be < steps_per_epoch
    epoch_period: 1
    logdir: ${loggers.logdir}
```

```yaml
# 00_bench/bench_base_ds.yaml -- epoch mode; launcher sets max_rows = 60 x bs_per_gpu x 8
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/00_common/base_study_ds
  - _self_
experiment_name: sam2_bench_ds
wandb_project: sam2_study_bench
datasets: {max_rows: 480, timepoint_list: [0]}
schedulers: {epochs: 1}
hooks:
  hooks_list:
  - name: free_device_buffer
  - name: lr_scheduler
    backend: DEEPSPEED
  - name: iteration_timer
  - name: periodic_writer
  - name: torch_memory_stats
    step_period: 10
    epoch_period: 1
    logdir: ${loggers.logdir}
```

```yaml
# 00_bench/bench_tile_base.yaml
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/00_bench/bench_base
  - _self_
experiment_name: sam2_bench_tile
clusters: {batch_size: 8}
datasets:
  max_rows: 512
  buffer_capacity: 4          # 2.6 GB host slot per tile
  databases: {sample_type: tile, max_tile_shape: [128, 512, 3328]}
```

One leaf per shape (no YAML anchors: they do not survive `OmegaConf.to_container` on `runs`):

```yaml
# 00_bench/shapes/crop_128x384x512.yaml
# crop_128x128x128 / crop_128x256x256 / crop_128x384x1024: same file with the three shape lines changed
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/00_bench/bench_tile_base
  - _self_
experiment_name: sam2_bench_crop_128x384x512
datasets:
  train_shape: [1, 128, 384, 512, 5]
  input_shape: [1, 128, 384, 512, 6]
  preprocessor:
    transforms_list:
      - _target_: cell_observatory_platform.data.transforms.crop.Crop
        target_spatial_shape: [128, 384, 512]
        crop_dims: YX
        crop_type: random
        bbox_format: ${datasets.collate_fn.bbox_output_format}
        boxes_normalized: ${datasets.collate_fn.normalize_bboxes}
        seed: ${seed}
      - _target_: cell_observatory_platform.data.transforms.normalize.Normalize
        input_layout:
          _target_: cell_observatory_platform.data.data_shapes.MULTICHANNEL_HYPERCUBE
          value: ${dataset_layout_order}
```

```yaml
# 00_bench/shapes/resize_128x384x1024.yaml -- today's path
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/00_bench/bench_tile_base
  - _self_
experiment_name: sam2_bench_resize_128x384x1024
datasets:
  train_shape: [1, 128, 384, 1024, 5]
  input_shape: [1, 128, 384, 1024, 6]
  preprocessor:
    transforms_list:
      - _target_: cell_observatory_platform.data.transforms.resize.Resize
        target_spatial_shape: [128, 384, 1024]
        mode: trilinear
        align_corners: false
        dtype: ${quantization}
        bbox_format: ${datasets.collate_fn.bbox_output_format}
        boxes_normalized: ${datasets.collate_fn.normalize_bboxes}
        crop_to_valid: true
      - _target_: cell_observatory_platform.data.transforms.normalize.Normalize
        input_layout:
          _target_: cell_observatory_platform.data.data_shapes.MULTICHANNEL_HYPERCUBE
          value: ${dataset_layout_order}
```

Launchers (`S=experiments/janelia/2026_09_02/sam2_study/00_bench`):

```yaml
# 00_bench/E1_backend.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E1
wandb_tags: ["sam2_study", "bench", "E1"]
sweep:
  - datasets: {preprocessor: {max_masks: [16, 46]}}
runs:
  - {cfg: S/bench_base.yaml,    name: torch_bs1.yaml, overrides: {clusters: {batch_size: 8}}}
  - {cfg: S/bench_base.yaml,    name: torch_bs2.yaml, overrides: {clusters: {batch_size: 16}}}
  - {cfg: S/bench_base_ds.yaml, name: ds_bs1.yaml,    overrides: {clusters: {batch_size: 8},  datasets: {max_rows: 480}}}
  - {cfg: S/bench_base_ds.yaml, name: ds_bs2.yaml,    overrides: {clusters: {batch_size: 16}, datasets: {max_rows: 960}}}
```

```yaml
# 00_bench/E2_mode_shape.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E2
wandb_tags: ["sam2_study", "bench", "E2"]
sweep:
  - datasets: {preprocessor: {max_masks: [46, 100]}}
runs:
  - {cfg: S/bench_base.yaml,                     name: cube128.yaml,        overrides: {clusters: {batch_size: 8}}}
  - {cfg: S/shapes/crop_128x128x128.yaml,        name: crop128.yaml,        overrides: {}}
  - {cfg: S/shapes/crop_128x256x256.yaml,        name: crop256.yaml,        overrides: {}}
  - {cfg: S/shapes/crop_128x384x512.yaml,        name: crop384x512.yaml,    overrides: {}}
  - {cfg: S/shapes/crop_128x384x1024.yaml,       name: crop384x1024.yaml,   overrides: {}}
  - {cfg: S/shapes/resize_128x384x1024.yaml,     name: resize384x1024.yaml, overrides: {}}
```

```yaml
# 00_bench/E3_bs_mm_cube128.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E3
wandb_tags: ["sam2_study", "bench", "E3"]
sweep:
  - clusters: {batch_size: [8, 16, 32, 64, 128, 256]}     # 1..32 per GPU
  - datasets: {preprocessor: {max_masks: [16, 46]}}
runs:
  - {cfg: S/bench_base.yaml, name: cube128.yaml, overrides: {}}
```

```yaml
# 00_bench/E4_bs_mm_crop384x512.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E4
wandb_tags: ["sam2_study", "bench", "E4"]
sweep:
  - clusters: {batch_size: [8, 16, 32]}
  - datasets: {preprocessor: {max_masks: [46, 100]}}
runs:
  - {cfg: S/shapes/crop_128x384x512.yaml, name: crop384x512.yaml, overrides: {datasets: {buffer_capacity: 8}}}
```

```yaml
# 00_bench/E5_compile.yaml -- BS_CUBE / BS_CROP = 8 x winners of E3 / E4
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E5
wandb_tags: ["sam2_study", "bench", "E5"]
sweep:
  - optimizations: {models: {sam: {torch_compile: {enable: [false, true]}}}}
runs:
  - {cfg: S/bench_base.yaml,              name: cube128.yaml,     overrides: {clusters: {batch_size: BS_CUBE}, trainer_loop: {max_steps: 100}}}
  - {cfg: S/shapes/crop_128x384x512.yaml, name: crop384x512.yaml, overrides: {clusters: {batch_size: BS_CROP}, trainer_loop: {max_steps: 100}, datasets: {preprocessor: {max_masks: 100}}}}
  - {cfg: S/bench_base_ds.yaml,           name: ds_cube128.yaml,  overrides: {clusters: {batch_size: BS_CUBE}, datasets: {max_rows: "100 x BS_CUBE"}}}
```

```yaml
# 00_bench/E6_fp8.yaml -- explicit runs (recipe x filter is not a product)
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E6
wandb_tags: ["sam2_study", "bench", "E6"]
runs:
  - {cfg: S/bench_base.yaml, name: cube_bf16.yaml,  overrides: {clusters: {batch_size: BS_CUBE}}}
  - {cfg: S/bench_base.yaml, name: cube_mxfp8.yaml, overrides: {clusters: {batch_size: BS_CUBE},
      parallelism: {quantize: {enable: true, backend: mx, recipe: mxfp8_cublas, filter_fqns: []}}}}
  - {cfg: S/bench_base.yaml, name: cube_mxfp8_rceil.yaml, overrides: {clusters: {batch_size: BS_CUBE},
      parallelism: {quantize: {enable: true, backend: mx, recipe: mxfp8_cublas_rceil, filter_fqns: []}}}}
  - {cfg: S/bench_base.yaml, name: cube_mxfp8_keepheads.yaml, overrides: {clusters: {batch_size: BS_CUBE},
      parallelism: {quantize: {enable: true, backend: mx, recipe: mxfp8_cublas,
                               filter_fqns: ["iou_prediction_head", "output_hypernetworks_mlps"]}}}}
  # same four on S/shapes/crop_128x384x512.yaml with clusters.batch_size: BS_CROP, max_masks 100
```
FP8 loss-parity check for the winner: 2 epochs of `01_lr/lr_base.yaml` + the quantize block vs bf16.

```yaml
# 00_bench/E7_fit_levers.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/00_bench/E7
wandb_tags: ["sam2_study", "bench", "E7"]
sweep:
  - optimizations: {models: {sam: {activation_checkpoint: {enable: [false, true]}}}}
  - models: {meta_arch: {sam: {use_act_ckpt_iterative_pt_sampling: [false, true]}}}
runs:
  - {cfg: S/shapes/crop_128x384x1024.yaml, name: crop384x1024.yaml, overrides: {datasets: {preprocessor: {max_masks: 100}}}}
```

```bash
# scripts/utils/training.sh addition -- dmon sidecar when BENCH_DMON=1 (set via optimizations.env in bench_base)
if [ "${BENCH_DMON:-0}" = "1" ]; then
  mkdir -p "$OUTDIR/dmon"
  nvidia-smi dmon -s um -d 5 -o T > "$OUTDIR/dmon/$(hostname).csv" 2>/dev/null &
  trap 'kill $! 2>/dev/null' EXIT
fi
```

```python
# scripts/utils/bench_table.py -- python bench_table.py $DATA_DIR/sam2_study/00_bench
import sys, glob, pandas as pd
rows = []
for f in glob.glob(f"{sys.argv[1]}/**/logs/step_logbook*.csv", recursive=True):
    d = pd.read_csv(f); d = d[d["step"] >= 20]
    cols = [c for c in d.columns if any(k in c for k in ("step_time", "data_time", "max_reserved", "tps", "mfu"))]
    rows.append({"run": f.split("/")[-3], "steps": len(d), **d[cols].median().round(3).to_dict()})
print(pd.DataFrame(rows).sort_values("run").to_markdown(index=False))   # steps < 40 => died early (OOM)
```

---

## 2. Step 1 — learning rate

One LR at the Stage 0 batch and shape. Current base LR is 1e-4 (all 09-01 bring-up variants sweep
5e-5 / 1e-4 / 2e-4 around it); sweep one decade around it.

- **Budget**: `S · E / (bs_global · samples_per_s) ≲ 3.2 h` (one 4 h job). Placeholder S = 65,536
  cubes at timepoint stride 10, E = 10. Same `seed`, `max_rows`, `timepoint_list`, `split` in every
  run → identical train/val rows.
- **Signal**: val `step_loss` per epoch (`best_checkpointer` + `best_metric_saver`) for sanity;
  **mask mAP on the held-out tiles** (`eval/test_heldout.yaml`) on each best checkpoint picks the winner.
- **Runs**: `lr_sweep` lr {5e-5, 1e-4, 2e-4, 4e-4}; optional `seed_repeat` {43, 44} at the winner
  (noise floor Δ; without it use Δ = 0.01 mAP).

```yaml
# 01_lr/lr_base.yaml -- fixed subset + validation + best checkpoint; one 4h job per run
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/00_common/base_study
  - _self_
experiment_name: sam2_lr
clusters: {batch_size: BS_CUBE}                 # from E3
datasets:
  split: 0.1
  max_rows: 65536
  timepoint_list: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                   130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
schedulers: {epochs: 10, warmup: 1, cooldown: 0, cos_min_ratio: 0.05}   # group default 0.5 only halves LR
optimizers: {lr: 1e-4, wd: "${eval:'${optimizers.lr} * 0.1'}"}
evaluation: {val_begin: 0, val_interval: 1}
trainer_loop: {iteration_mode: epoch, checkpoint_every_n_epochs: 1, with_perf_metrics: false}
hooks:
  hooks_list:
  - name: free_device_buffer
  - name: lr_scheduler
    backend: TORCHTITAN
  - name: iteration_timer
  - name: periodic_writer
  - name: periodic_checkpointer
    file_prefix: latest_model
    backend: TORCHTITAN
    time_interval: 3h30m
  - name: best_checkpointer
    checkpointdir: ${checkpoint.checkpoint_manager.save_checkpointdir}
  - name: best_metric_saver
    metric_name: ${evaluation.val_metric}      # step_loss
    compare_fn: ${evaluation.val_mode}         # min
    eval_after_validation: true
```

```yaml
# 01_lr/lr_sweep.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/01_lr
wandb_tags: ["sam2_study", "step1", "lr"]
sweep:
  - optimizers: {lr: [5e-5, 1e-4, 2e-4, 4e-4]}
runs:
  - {cfg: experiments/janelia/2026_09_02/sam2_study/01_lr/lr_base.yaml, name: lr.yaml, overrides: {}}
```

```yaml
# 01_lr/seed_repeat.yaml -- NOTE: seed also seeds the train/val split; needs datasets.split_seed (v1 C5) for a fixed split
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/01_lr
wandb_tags: ["sam2_study", "step1", "seed"]
sweep:
  - seed: [43, 44]
runs:
  - {cfg: experiments/janelia/2026_09_02/sam2_study/01_lr/lr_base.yaml, name: seed.yaml, overrides: {optimizers: {lr: LR_WIN}}}
```

```yaml
# eval/test_heldout.yaml -- mask mAP on tile-disjoint held-out cubes, from a run's best checkpoint
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/01_lr/lr_base
  - override /evaluation: sam2_instance_evaluator
  - _self_
experiment_name: sam2_eval_heldout
job_type: test
paths: {resume_checkpointdir: RUN_OUTDIR/checkpoints}     # best_checkpointer dir of the run under test
datasets:
  split: 0.0
  max_rows: null
  timepoint_list: [0, 50, 100, 150, 200]
  roi_ids: HELDOUT_TILES                                  # 4 tiles by roi_id, excluded from every training subset (D3)
hooks:
  hooks_list:
  - name: free_device_buffer
  - name: periodic_writer
```

---

## 3. Step 2 — channel parametrization, channel dropout, interaction protocol

Greedy on the Step 1 recipe, judged on held-out mAP vs Δ. Order A → B → D (B runs on the A winner;
D is independent and runs in parallel with B).

**A. Channel parametrization** (how the 5 channels enter the ViT; code sketch v1 C6):

| id | variant | cost |
|---|---|---|
| A0 | joint linear over the 5-channel patch (current) | 1× |
| A1 | per-channel tokens + channel embedding, `attn_pool` fusion | ≈1× |
| A2 | per-channel tokens, `concat` fusion | 5× backbone; only if A1 helps, bench first |
| A3 | membrane-only (C=1), cytosol-only (C=4) | ≤1×, config only |

**B. Channel dropout** (transform, v1 C7): zero a random subset of the 4 cytosol channels, membrane
kept, p {0.25, 0.5} × shuffle {F, T}. With A1 it can *drop* channels (variable C): test last if A1 wins.

**D. Interaction**: `num_correction_pt_per_frame` {0, 3} × `prob_to_sample_from_gt_for_train`
{0.0, 0.3} (7 / 0.0 = incumbent); box prompts p 0.5 (needs B10: box format from collator +
denormalize). Fewer correction rounds is also the biggest memory lever → feeds back into §1.4.

| launcher | points | prereq |
|---|---|---|
| `02_abl/channel_param.yaml` | A1, A2, A3-membrane, A3-cytosol | C6 |
| `02_abl/channel_dropout.yaml` | 4 | C7 |
| `02_abl/correction.yaml` | 4 | — |
| `02_abl/box_prompt.yaml` | 1 | B10 |

```yaml
# 02_abl/abl_base.yaml -- Step 1 winner, frozen
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/01_lr/lr_base
  - _self_
experiment_name: sam2_abl
optimizers: {lr: LR_WIN}
```

```yaml
# 02_abl/channel_param.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/02_abl
wandb_tags: ["sam2_study", "step2", "channel_param"]
runs:
  - cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml
    name: A1_chan_tokens_attnpool.yaml
    overrides:
      models:
        backbones:
          masked_encoder:
            patch_embed_type: channel_adaptive        # new key (v1 C6)
            patch_embed_args: {max_channels: 8, use_channel_embed: true, channel_fusion: attn_pool, attn_pool_num_heads: 8}
  - cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml
    name: A2_chan_tokens_concat.yaml
    overrides:
      models:
        backbones:
          masked_encoder:
            patch_embed_type: channel_adaptive
            patch_embed_args: {max_channels: 8, use_channel_embed: true, channel_fusion: concat}   # 5x tokens
  - cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml
    name: A3_membrane_only.yaml
    overrides:
      datasets:
        train_shape: [1, 128, 128, 128, 1]
        input_shape: [1, 128, 128, 128, 2]
        selected_channel_localizations: ["membrane"]
        databases: {required_channel_localizations: ["membrane", "cytosol"], data_channel_count: 5}   # same ROI set
  - cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml
    name: A3_cytosol_only.yaml
    overrides:
      datasets:
        train_shape: [1, 128, 128, 128, 4]
        input_shape: [1, 128, 128, 128, 5]
        selected_channel_localizations: ["cytosol"]
        databases: {required_channel_localizations: ["membrane", "cytosol"], data_channel_count: 5}
```

```yaml
# 02_abl/channel_dropout.yaml -- sweep indexes transforms_list[1]; confirm the dotlist merge lands there on first submit
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/02_abl
wandb_tags: ["sam2_study", "step2", "channel_dropout"]
sweep:
  - datasets: {preprocessor: {transforms_list: {1: {p: [0.25, 0.5]}}}}
  - datasets: {preprocessor: {transforms_list: {1: {shuffle: [false, true]}}}}
runs:
  - cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml
    name: chan_drop.yaml
    overrides:
      datasets:
        preprocessor:
          transforms_list:
            - _target_: cell_observatory_platform.data.transforms.normalize.Normalize
              input_layout:
                _target_: cell_observatory_platform.data.data_shapes.MULTICHANNEL_HYPERCUBE
                value: ${dataset_layout_order}
            - _target_: cell_observatory_platform.data.transforms.channel_dropout.ChannelDropout   # v1 C7
              p: 0.25
              shuffle: false
              always_keep: [0]          # membrane
              mode: zero                # zero | drop (drop only with patch_embed_type: channel_adaptive)
              seed: ${seed}
```

```yaml
# 02_abl/correction.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/02_abl
wandb_tags: ["sam2_study", "step2", "correction"]
sweep:
  - models: {meta_arch: {sam: {num_correction_pt_per_frame: [0, 3]}}}
  - models: {meta_arch: {sam: {prob_to_sample_from_gt_for_train: [0.0, 0.3]}}}
runs:
  - {cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml, name: corr.yaml, overrides: {}}
```

```yaml
# 02_abl/box_prompt.yaml -- needs B10
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/02_abl
wandb_tags: ["sam2_study", "step2", "box"]
runs:
  - cfg: experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base.yaml
    name: box_p0p5.yaml
    overrides:
      models: {meta_arch: {sam: {prob_to_use_box_input_for_train: 0.5, prob_to_use_box_input_for_eval: 0.0}}}
```

---

## 4. Step 3 — scale: epochs, pretrained init, ViT-H

On the Step 2 recipe, chained across 4 h jobs (`sam2_chain.sh` below). Compile / MX-FP8 on if they
won in Stage 0 (FP8 after the parity check).

| id | run | change | notes |
|---|---|---|---|
| S1 | more epochs | E × 2, E × 4; plus same steps on more data (stride 5, E = 1) | keep the full-data run if × 4 overfits |
| S2 | pretrained init | `pretrained_checkpointdir` = MAE/JEPA ViT-L, key remap (v1 C9); freeze {none, full, last 8 blocks} | at S1's best schedule |
| S3 | ViT-H | `masked_encoder/huge`, `backbone_native_channels: 1280`, LR × 0.5 | bench E3 at bs {1, 2, 4} first |

```yaml
# 03_scale/scale_base.yaml
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/02_abl/abl_base
  - _self_
experiment_name: sam2_scale
schedulers: {epochs: 20, warmup: 1, cos_min_ratio: 0.05}
evaluation: {val_interval: 1}
```

```yaml
# 03_scale/S1_epochs.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/03_scale
wandb_tags: ["sam2_study", "step3", "epochs"]
runs:
  - {cfg: experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base.yaml, name: ep_x2.yaml, overrides: {schedulers: {epochs: 20}}}
  - {cfg: experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base.yaml, name: ep_x4.yaml, overrides: {schedulers: {epochs: 40}}}
  - cfg: experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base.yaml
    name: fulldata_ep1.yaml            # same optimizer steps as ep_x4
    overrides:
      datasets:
        max_rows: null
        split: 0.02
        timepoint_list: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                         100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
                         180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240]
      schedulers: {epochs: 1, warmup: 0.1}
```

```yaml
# 03_scale/S2_pretrained.yaml -- KEYMAP = {"masked_encoder.encoder.": "image_encoder.backbone.encoder."} (v1 C9)
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/03_scale
wandb_tags: ["sam2_study", "step3", "pretrained"]
runs:
  - {cfg: experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base.yaml, name: init_mae_full.yaml,
     overrides: {paths: {pretrained_checkpointdir: MAE_CKPT}, checkpoint: {checkpoint_manager: {pretrained_key_map: KEYMAP}}}}
  - {cfg: experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base.yaml, name: init_mae_frozen.yaml,
     overrides: {paths: {pretrained_checkpointdir: MAE_CKPT}, checkpoint: {checkpoint_manager: {pretrained_key_map: KEYMAP}},
                 models: {meta_arch: {sam: {freeze_image_encoder: true}}}}}
  - {cfg: experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base.yaml, name: init_mae_last8.yaml,
     overrides: {paths: {pretrained_checkpointdir: MAE_CKPT}, checkpoint: {checkpoint_manager: {pretrained_key_map: KEYMAP}},
                 models: {backbones: {sam_backbone: {blocks_to_train: ["16","17","18","19","20","21","22","23"]}}}}}
```

```yaml
# 03_scale/S3_vit_h.yaml -- /optimizations/models/sam/large applies unchanged (block paths are size-independent)
# @package _global_
defaults:
  - /experiments/janelia/2026_09_02/sam2_study/03_scale/scale_base
  - override /models/backbones/masked_encoder: huge
  - _self_
run_type: single_run
experiment_name: sam2_scale_vit_h
models: {backbones: {sam_backbone: {backbone_native_channels: 1280}}}
optimizers: {lr: "LR_WIN x 0.5"}
```

```bash
# scripts/utils/sam2_chain.sh <cfg> <n_jobs> -- N dependent 4h LSF jobs; job k>1 resumes from the run's checkpoints
CFG=$1; N=${2:-4}; DEP=""
for k in $(seq 1 $N); do
  OV=""; [ $k -gt 1 ] && OV="checkpoint.resume_run=true paths.resume_checkpointdir=\${paths.outdir}/checkpoints"
  JID=$(python3 manager.py --config-name=$CFG clusters.dependency="$DEP" $OV | grep -o 'Job <[0-9]*>' | tr -dc 0-9)
  DEP="done($JID)"; echo "job $k -> $JID"
done
```

---

## 5. Open decisions

- **D1** E2 shapes: 128³ / 256² / 384×512 / 384×1024 crops + resize 1024. Drop 128×256×256 for a shorter matrix?
- **D2** seed repeat (2 jobs) or fixed Δ = 0.01 mAP.
- **D3** held-out tiles: 4 by `roi_id`, excluded from every training subset (needs v1 C4 `exclude_tile_list`, or `roi_ids` allow-lists). Pick them, or I take the 4 highest.
- **D4** FP8 parity gate: loss within noise over 2 epochs, or one extra mAP job.
- **D5** DeepSpeed in E5: kept to separate compile gain from compile × FSDP2; drop if E1 retires DeepSpeed.

## 6. Tests

| id | test | covers |
|---|---|---|
| T1 | 1-GPU smoke of `bench_base` (`/clusters/local`, `max_steps: 3`, `max_rows: 16`, mm 8): 3 steps, perf + memory columns present in `step_logbook` | base, hooks |
| T2 | same on `bench_base_ds` (`max_rows: 3`, `epochs: 1`) | DS twin |
| T3 | `manager.py` dry-run of every launcher: materialized yaml count = §1.2, unique `experiment_name`/outdir | launchers |
| T4 | `bench_table.py` on two synthetic run dirs (one dies at step 30): early row flagged, clean row uses steps 20–60 | table |
| T5 | `quantize.enable: true, backend: mx` on non-SM100 raises at construction | E6 cannot silently run bf16 |
| T6 | compile on, 2 steps, DCP save, resume with compile off → identical state-dict keys | E5 rule |
| T7 | `ChannelDropout`: channel 0 never zero; `p=1.0` zeroes all droppable; `shuffle` is a permutation | C7 |
| T8 | `MaskedEncoder(patch_embed_type=channel_adaptive)`: `attn_pool` → 512 tokens, `concat` → 2560; unpatchify `(1,1024,8,8,8)` | C6 |
| T9 | `job_type: test` on 4 held-out cubes, random init → all metrics finite | eval |
| T10 | existing suite green | regression |
