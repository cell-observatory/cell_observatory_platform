# Stage 1.5 — tile-scale adaptation at 128×384×1024

2026-09-03 · status: **PROPOSAL; bench row landed: bs 4/GPU, 2.00 s/step, 237 GiB (86 %), 16 samples/s/node** · configs: `configs/experiments/janelia/tests/2026_09_03/sam2_tile/` (to write once the bs row lands) · recipe: `…/2026_09_02/sam2/recipe_r1.yaml`

## 1. What we are investigating and why

Stage 1–3 train on 128×384×512 cubes. Inference has to run on whole tiles (128×512×{2560…3328}), i.e. windows several
times wider than anything the model saw in training. The ViT uses absolute sincos positions (`abs_sincos_enc`,
per-patch index), so a 512-wide checkpoint loads into a 1024-wide model unchanged except the pos-embed buffer, which is
re-initialised at the new shape (`training/checkpoint.py` drops the mismatched tensor; the values for X positions 0–31
are identical). What the model has never done is attend over 64 X-patches (12 288 tokens) and prompt masks in the wider
context. Question: **does a short fine-tune at 128×384×1024 (the largest DB-side cube = ½ a tile in X) make tile-level
inference work from the existing checkpoint, at ~10 % of the cube run's compute?**

Evidence:

| fact | value | source |
|---|---|---|
| rows at 128×384×1024, annotated | **141,901** (+ 9,205 unannotated) | live DB count 2026-09-03 19:00 (the fringe fix removed the out-of-bounds cubes; the stage-0c doc's 314,203 was the pre-fix total) |
| recipe at 128×384×512, bs 8/GPU | 1.68 s/step, 237 GiB, 38 samples/s/node | perf summary |
| old (7-click) code at 128×384×1024, bs 1 | 1.96 s/step, 215 GiB; same voxel rate as c512 | stage-0c K1b |
| Stage-1 budget (one lr run) | 10 epochs × 247 k = **2.5 M cube samples**, ~18 h/node | lr plan (corrected count) |

## 2. Runs

Variable: **none** (one adaptation run per Stage-1 winner); the two-way comparison is against the unadapted checkpoint.

| run | init | shape | budget | rationale |
|---|---|---|---|---|
| `tile_adapt` | Stage-1 best checkpoint (`paths.pretrained_checkpointdir` + identity `pretrained_key_map` so the load is permissive) | 128×384×1024 cubes, all 142 k rows | 10 % of Stage 1 = 0.25 M c512-sample-equivalents = **125 k c1024 samples (0.9 epoch)** | enough steps to learn the wider context, cheap enough to repeat per checkpoint |
| `tile_adapt_from_scratch` (optional) | random init | same | same | only if `tile_adapt` beats the zero-shot checkpoint by less than the noise floor — tells whether the 512 pretraining transfers at all |

Held constant (recipe, `recipe_r1.yaml`):

| | value | why |
|---|---|---|
| hardware | 1 B300 node, FSDP2, chained 4 h jobs | same as Stage 1 |
| recipe | 1 click, low-res click loop, GEMM up/down-scaling, criterion ckpt off, `mask_rows: global_max`, `cudnn_benchmark: false` | Stage-0 winner |
| batch | **bs 4/GPU → global 32** (`sweep/sam2_c1024_bs4`: 237 GiB = 86 %, 0 retries; bs 5 will not fit) | VRAM law: 2× voxels of c512 → ½ its bs |
| masks | `max_masks: 96` (cap; a 1024-wide cube holds ~2× the instances of a 512 one, cube max 46 → ~90) | mm ≤ 100 rule |
| lr | Stage-1 winner × 0.5, warm-up 5 %, cosine to 5 % | adaptation, not training from scratch |
| data | shape-7 rows, `split: 0.02`, `in_bounds_only`; PSF + sensor-noise transforms from the recipe (the PSF transform re-prepares its OTF for the 1024-wide shape) | all data, as requested |
| tile path alternatives (measured 2026-09-03) | `sample_type: tile` crop 128×384×1024: bs 2, 1.74 s/step, 214 GiB, 96 masks; resize: 1.83 s, 215 GiB; neither I/O-bound any more (≈1.1 samples/s/GPU vs 2.0 on DB cubes at 48 masks) | DB cubes stay the training path; the tile path is viable for eval and for a 96-mask variant |

Budget at the measured **16 samples/s/node** (2.0 samples/s/GPU):

| c1024 samples | epochs of 142 k | hours at 16 samples/s | share of Stage 1 (18 h) |
|---|---|---|---|
| 70 k | 0.5 | 1.2 | 7 % |
| **125 k** | **0.9** | **2.2** | **12 %** |
| 139 k (1 epoch, 2 % val) | 1.0 | 2.4 | 13 % |

One 4 h job (`chain_jobs: 2` as the crash guard) covers a full epoch (2.4 h + validation); the run is **1 epoch ≈ 13 % of Stage 1**.

## 3. How we evaluate

- **Primary: mask mAP on the held-out tiles at full tile size** (`eval/test_heldout` on `sample_type: tile`, resize-free
  crop of 128×384×W per tile, sliding in X), three checkpoints: Stage-1 best evaluated zero-shot at 1024, `tile_adapt`,
  and Stage-1 best evaluated at 512 (its native shape, the reference).
- Secondary: val loss at 1024 during the run (should drop below the zero-shot value within the first 10 % of steps);
  held-out mAP at 512 after adaptation (must not regress by more than Δ).
- Noise floor Δ = 0.01 mAP (Stage-1 convention). Decision: `tile_adapt` wins if it beats zero-shot by > Δ at tile size
  and holds 512 mAP within Δ; then every later production checkpoint gets the same 10 % adaptation pass. If it loses at
  512, try lr × 0.25; if it does not beat zero-shot, run `tile_adapt_from_scratch` before concluding.

Prerequisites (gates): (a) bench row `sam2_c1024_bs*` → BS_1024, RATE; (b) pretrained load of a 512 checkpoint into the
1024 model drops exactly `pos_embed` (check the `[CheckpointManager] Dropped` log line on a 20-step smoke); (c) tile-mode
evaluation config with X-sliding windows (tile rows are 128×512×{2560…3328}; the loader's tile path is I/O-bound at
~20 s/sample/rank on /groups, acceptable for a 4-tile eval, not for training — hence DB-side cubes for training).

## 4. Data

| run | init | bs/GPU | s/step | samples/s/node | GiB | steps | val loss (end) | mAP tile (zero-shot → adapted) | mAP 512 (before → after) | wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `sweep/sam2_c1024_bs4` (bench) | – | 4 | 2.00 | 16 | 237 | 200 | – | – | – | 9 min |
| `tile_adapt` | Stage-1 best | BS_1024 | | | | | | | | |

Reflections: (fill after the runs)

## 5. Config sketch

```yaml
# configs/experiments/janelia/tests/2026_09_03/sam2_tile/tile_adapt.yaml
# @package _global_
defaults:
  - /experiments/janelia/tests/2026_09_02/sam2/recipe_r1
  - _self_
experiment_name: sam2_tile_adapt
wandb_project: sam2_study
paths:
  pretrained_checkpointdir: STAGE1_BEST/checkpoints          # best_checkpointer dir of the Stage-1 winner
checkpoint: {checkpoint_manager: {pretrained_key_map: {"image_encoder": "image_encoder"}}}   # identity map -> permissive load; only pos_embed is dropped
datasets:
  train_shape: [1, 128, 384, 1024, 5]
  input_shape: [1, 128, 384, 1024, 6]
  split: 0.02
  buffer_capacity: 12
  preprocessor: {max_masks: 96}
trainer_loop: {iteration_mode: epoch, checkpoint_every_n_epochs: 1, with_perf_metrics: false}
schedulers: {epochs: 1, warmup: 0.05, cooldown: 0, cos_min_ratio: 0.05}
optimizers: {lr: "LR_WIN * 0.5", wd: "${eval:'${optimizers.lr} * 0.1'}"}
evaluation: {val_begin: 0, val_interval: 1}
clusters: {batch_size: 32, timelimit: "04:00", chain_jobs: 2}
loggers: {event_writers: [{name: scalars}, {name: wandb, id: "${experiment_name}", resume_from: allow}]}
```
