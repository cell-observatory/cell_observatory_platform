# Stage 1 — learning-rate sweep

2026-09-03 · configs: `configs/experiments/janelia/tests/2026_09_03/sam2_lr/{lr_base,lr_sweep,chain_smoke}.yaml` · recipe: `…/2026_09_02/sam2/recipe_r1.yaml` (bs 8/GPU)

## 1. What we are investigating and why

The learning rate has to be settled before the Stage-2 ablations (channel parametrisation, click protocol), otherwise every
ablation confounds "worse lr" with "worse setting". The training recipe changed enough from the last good run that the old
value cannot be assumed: larger cubes (128×384×512 vs 128×256×512), 48 masks instead of 40, 1 simulated click instead of 7,
the low-res click loop, and FSDP2 instead of DeepSpeed. The global batch is close (64 vs 40), which is what the lr scales
with, so the old run is direct evidence.

Evidence from the old run (`sam2_baseline_128x256x512_150_epoch_lr_0p0001_opt_adamw`, W&B `output.log`):

| setting | old run |
|---|---|
| hardware | 5 nodes × 8 H100 80 GB, DeepSpeed ZeRO-2 |
| global batch | 40 (1/GPU) · 439 steps/epoch → 17.6 k samples/epoch |
| budget | 150 epochs planned, 142 logged → **2.5 M samples** |
| lr | **1e-4** worked (train loss 2.72 at epoch 142); the twin at **1e-3 diverged** (21.7 at epoch 89), same batch |
| schedule | AdamW (wd = 0.1·lr), warm-up 30 epochs (20 %), cosine |

So 1e-4 is proven and 1e-3 is a proven failure at this batch; the probes go between them, aggressive first.

## 2. Runs

Variable: **learning rate only**. Three runs, one B300 node each, submitted in parallel.

| run | lr | rationale |
|---|---|---|
| `lr_2e-4` | 2e-4 | 2× the proven value |
| `lr_4e-4` | 4e-4 | upper probe: 2.5× below the known divergence |
| `lr_1e-4` | 1e-4 | anchor (proven) |

Follow-ups, decided from the data section: if 4e-4 wins → `lr_6e-4`; if both probes lose to the anchor → `lr_5e-5`, `lr_2e-5`.

Held constant (the Stage-0 recipe, `recipe_r1.yaml`):

| | value | why |
|---|---|---|
| hardware | 1 B300 node (8 GPUs, 275 GB each), torch-native FSDP2, 4 h LSF slots chained (`clusters.chain_jobs`) | one node per run; runs in parallel |
| recipe | 1 click, low-res click loop + low-res IoU, GEMM+shuffle up/down-scaling, criterion act-ckpt off, `mask_rows: global_max`, `cudnn_benchmark: false` | measured fastest setup at bs 8: **1.68 s/step, 237 GiB (86 %), 4.77 samples/s/GPU = 38 samples/s/node** (bs 7: 4.38; bs 10 + decoder act-ckpt OOMs); time is ~41 % backward, ~19 % decoder passes, ~18 % ViT, the rest per-mask sampling/loss (memory-bound, not GEMM-bound) |
| batch | 8/GPU → **global 64** | fills VRAM (86 %); same regime as the old run's 40 (1.6×) |
| masks | `max_masks: 48`, `in_bounds_only` | Stage-0 setting |
| image formation (added 2026-09-08) | the synthetic cubes are clean rendered intensities, so the recipe now applies PSF convolution (measured DSR light-sheet PSF `CamA_Arc_OMW`, centred + uncompressed copy at `$DATA_DIR/psf/CamA_Arc_OMW_centered.tif`, 0.1 µm voxels assumed for PSF and data) → mixed Poisson-Gaussian sensor noise (QE 0.82, 0.22 e⁻/count, read noise 40, offset 100 counts, `tasks/denoising.yaml`) → Normalize, on the 5 signal channels only (the mask is split off first). Measured cost on the recipe: none (1.57 s/step at bs 8 with the Gaussian stand-in; real-PSF trial `sweep/sam2_realpsf_trial` 2026-09-08). Inspection PNGs: `$DATA_DIR/sam2_study/psf_noise_check_realpsf/` | make synthetic training data resemble real acquisitions (Hugo 2026-09-03/08) |
| excluded tiles | `exclude_tile_path_patterns`: `20250311_mem_histone/fish1_24hpf/roi4/000x_001y_000z.zarr`, `20250528_mem-histone/fish3/roi2_spine/000x_009y_001z.zarr` (unreadable shards found by `scripts/utils/scan_zarr_chunks.py`, 2,650 of 4,586 tiles scanned) + loader `read_failure_policy: zero_fill` as the safety net | the first sweep attempt died on a corrupt chunk every link |
| data | fixed 2026-09-03 snapshot, cube 128×384×512, **full set: 252,431 annotated cubes** (live DB count 2026-09-03 19:00: the fringe fix *removed* the out-of-bounds cubes, so the earlier 507,538 figure was the pre-fix total; + 91,177 unannotated cubes of this shape exist for pretraining), `split: 0.02` → ~247 k train / ~5 k val | full dataset as requested; 2 % val keeps validation ≈ 3 min/epoch |
| epochs | **20**, warm-up 4 (20 %, as the old run), cosine to 5 %, `max_norm 1`, wd = 0.1·lr | see budget table; Hugo picked 20 on 2026-09-03 after the cube count was corrected |
| seed / subset | same `seed`; identical rows for every run | only lr differs |

Epoch = 247 k / 38 samples/s = **1.8 h**. Budget options (per run, wall-clock; `chain_jobs: 14` covers 20 epochs with spares):

| epochs | samples | × old budget | wall-clock per run |
|---|---|---|---|
| 5 | 1.2 M | 0.5× | 9 h |
| 10 | 2.5 M | 1.0× | 18 h |
| **20** | 4.9 M | 2.0× | **1.5 days** |
| 30 | 7.4 M | 3.0× | 2.3 days |

With the corrected row count 10 epochs = the old run's sample budget (each sample is a 3× larger cube) in 18 h per run;
20 epochs (2×, 1.5 days) is the alternative if Hugo wants the doubled budget the earlier draft promised. Stage-2 ablations
reuse the winner's schedule. (`schedulers.epochs` in `lr_base.yaml` is 10; change to 20 for the 2× option.)

## 3. How we evaluate

- Primary: **held-out mask mAP** on the tile-disjoint held-out cubes (`eval/test_heldout`, plan_sam2_v2 §2) at each run's
  best-val checkpoint (`best_checkpointer` on val `step_loss`).
- Secondary: val `step_loss` curve every epoch (early-stop signal; a run whose val loss rises for 3 epochs is stopped) and
  train-loss stability in the first 2 epochs (warm-up) — the 1e-3 failure showed up as loss growth, not NaNs.
- Noise floor: Δ mAP < 0.01 is a tie → prefer the lower lr. Optional `seed_repeat` (seeds 43, 44) at the winner if the top two tie.
- Decision rule: winner = best mAP; if the winner is the top probe (4e-4) run 6e-4 before choosing; if the anchor wins run the
  lower half-decade.

## 4. Data

Launched 2026-09-03 19:14 from the run worktree (tag `runs/sam2-stage1-2026-09-03`, commit 367e077), one chain each.
Outputs: `$DATA_DIR/sam2_study/sam2_lr/sam2_study/01_lr/lr_sweep_optimizers_lr_<lr>/` (the recipe's `paths.outdir` is
prepended to `data_base_dir` by multi_run, hence the doubled prefix). W&B `sam2_study`, run ids `lr_sweep_optimizers_lr_0p000{2,4,1}`.

| run | lr | first job | status | epochs done | train loss (last) | best val loss (epoch) | held-out mAP | wall-clock | notes |
|---|---|---|---|---|---|---|---|---|---|
| `lr_sweep_optimizers_lr_0p0002` | 2e-4 | 153964933 | **killed 19:44** (attempt 1) | 0 | | | | 25 min | corrupt zarr chunk, see below |
| `lr_sweep_optimizers_lr_0p0004` | 4e-4 | 153964934 | **killed 19:44** (attempt 1) | 0 | | | | 25 min | same |
| `lr_sweep_optimizers_lr_0p0001` | 1e-4 | 153964935 | **killed 19:44** (attempt 1) | 0 | | | | 25 min | same |
| attempt 2, all three | | 154088034/35/36 (09-08 14:17) | **died at ~2 h 45 min**, followers killed 17:25 | 0 | | | | ~3 h × 3 nodes | end-of-epoch-1 crash: validation collator starved of device-buffer slots → NCCL timeout (`setup_sweep.md` G8); no checkpoint (3.5 h wall-clock save) |
| attempt 3 | all | 154089019 (2e-4), 154089020 (4e-4), 154089021 (1e-4), 09-08 19:00 | submitted | | | | | | fixes 30726a7: `pt_sampling_for_eval: uniform` (root cause), own validation collator, hourly checkpoints; verified by `chain_smoke_mini_fix` (validation step 0.95 s, val loss 2.59 at epoch 0, checkpoint at the epoch end); worktree tag `runs/sam2-stage1-2026-09-08c` |

Attempt 1 (19:14–19:44): all three runs died at ~25 min on the same unreadable chunk
(`20250311_mem_histone/fish1_24hpf/roi4/000x_001y_000z.zarr/c/0/0/1/14/0`, tensorstore "Invalid blosc-compressed data";
the file has a normal 46 MB size, so it is corrupt content, not truncation). Ray Train aborts the run on a loader error, and
the chain resumed from scratch each time (no checkpoint yet), so the chains were stopped (`CHAIN_STOP`) and killed. Fixes on
the dev branch: `datasets.databases.exclude_tile_path_patterns` (SQL `NOT LIKE`, set in both recipes for this tile, tests in
`tests/data/test_local_metadata_store.py`); the LSF wrapper now records the chain exit before its self-`bkill` and propagates
the task's exit code; `scripts/utils/scan_zarr_chunks.py` scans a data root for further unreadable chunks (run on a node).

Reflections: epoch 1 (2026-09-08 21:30): all three train and validate cleanly; ordering 4e-4 < 2e-4 < 1e-4 on val loss, no divergence at 4e-4 (warm-up runs to epoch 4, so the aggressive lr has not yet been tested at full rate). GPU step 1.37 s but 2.1–2.3 s wall-clock per step: ~35 % of the time is the data path at full-dataset scale (the 200-step probes hid it). Validation step 0.95 s after the uniform-sampler fix.

## 5. Configs (written; compose-checked 2026-09-08 with the real PSF: shape 128×384×512, bs 64 = 8/GPU, 20 epochs, warm-up 4, lr {2e-4, 4e-4, 1e-4}, chain_jobs 14, transforms PSF→noise→Normalize, two excluded tiles)

Launch prerequisites (2026-09-08): the recipe/loader changes are uncommitted in the dev tree; the run worktree `run_parent/cell_observatory_platform` (tag `runs/sam2-stage1-2026-09-03`) must receive them by commit + `git merge --ff-only` + re-tag before submitting (see memory `run-worktree-strategy`).

```yaml
# configs/experiments/janelia/tests/2026_09_03/sam2_lr/lr_base.yaml
# @package _global_
defaults:
  - /experiments/janelia/tests/2026_09_02/sam2/recipe_r1
  - _self_
experiment_name: sam2_lr
wandb_project: sam2_study
datasets: {split: 0.02}                       # full 2026-09-03 snapshot, 128x384x512, in_bounds_only, mm 48 (from the recipe)
schedulers: {epochs: 10, warmup: 2, cooldown: 0, cos_min_ratio: 0.05}
optimizers: {lr: 1e-4, wd: "${eval:'${optimizers.lr} * 0.1'}"}
evaluation: {val_begin: 0, val_interval: 1}
trainer_loop: {iteration_mode: epoch, checkpoint_every_n_epochs: 1, with_perf_metrics: false}
clusters: {worker_nodes: 1, gpus_per_worker: 8, batch_size: 64, timelimit: "04:00", chain_jobs: 12}   # resubmit-if-not-done (cluster/chain_lib.sh)
loggers: {event_writers: [{name: scalars}, {name: wandb, id: "${experiment_name}", resume_from: allow}]}   # one W&B run across the chain (merge, keep the other wandb fields)
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
    metric_name: ${evaluation.val_metric}
    compare_fn: ${evaluation.val_mode}
    eval_after_validation: true
```

```yaml
# configs/experiments/janelia/tests/2026_09_03/sam2_lr/lr_sweep.yaml
# @package _global_
run_type: multi_run
data_base_dir: sam2_study/01_lr
wandb_tags: ["sam2_study", "step1", "lr"]
sweep:
  - optimizers: {lr: [2e-4, 4e-4, 1e-4]}
runs:
  - {cfg: experiments/janelia/tests/2026_09_03/sam2_lr/lr_base.yaml, name: lr.yaml, overrides: {}}
```

Also written: `chain_smoke.yaml` (recipe on 8 k rows, 4 epochs, 16 min limit, `chain_jobs: 3`, 5-min checkpoints) = the live
chaining test; run it before the sweep: `python manager.py --config-name=experiments/janelia/tests/2026_09_03/sam2_lr/chain_smoke.yaml`.

Before submitting: the resume gate (`stage0c/C1_resume_a` → `C1_resume_b`) must pass once on the recipe; the chaining
(`2026-09-03-lsf-chaining-sketch.md`, implemented, needs its LSF smoke test); `eval/test_heldout` must exist; `pt_sampling_for_eval` is
`uniform` in the recipes since 2026-09-08 (the `center` sampler stalled every validation epoch of attempt 2).
