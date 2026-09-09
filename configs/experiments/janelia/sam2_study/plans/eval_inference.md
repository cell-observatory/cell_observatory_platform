# Evaluation + inference of the Stage-1 checkpoints

2026-09-09 · configs `configs/experiments/janelia/sam2_study/eval/{eval_base,eval_lr_*,infer_base,infer_lr_*}.yaml` · inference block `configs/inference/inference_instance_segmentation_sam2.yaml` · run on the b300 node in local mode (8 GPUs)

## 1. What we are investigating and why

The Stage-1 lr decision (`stage1_lr_plan.md` §3) rests on held-out **mask mAP**, not on the validation loss: the training/validation
forward is *prompted* (one GT-interior click per object plus GT-driven corrections, all objects given), so its dice measures
delineation of known objects with perfect recall by construction. Inference is *unprompted* automatic mask generation (AMG: a
blind 8×8×8 click lattice, one decoder pass + one mask-to-mask refinement, IoU/stability/NMS filters), which can miss objects,
duplicate them or admit background. This pass (a) makes the prediction-based test loop and the plot-only inference loop work
end to end on the study recipe, and (b) gives a first AMG reading per lr at epoch 6 of 20.

| fact | value | source |
|---|---|---|
| checkpoints | epoch 6 (step 23076) of the three lr runs, copied to `$DATA_DIR/sam2_study/eval/ckpts/lr_<lr>/checkpoints` | trainer keeps 3 saves, deleted ~15:00 |
| val loss @6 (total / dice) | 2e-4 0.42 / 0.24 · 4e-4 0.48 / 0.28 · 1e-4 0.42 / 0.25 | `stage1_lr_plan.md` §4 |
| held-out rows | the runs' own 2 % validation split (~5 k cubes): same query, `seed: 42`, `split: 0.02` reproduce the split exactly (`datasets.eval_split: val`, new) | `SampleIndexPlanner.split_train_val` |
| tile-disjoint held-out tiles (plan D3) | not defined; the sweep trained on every tile | — |

## 2. Runs

Variable: **checkpoint (lr)**. Everything else fixed:

| | value | why |
|---|---|---|
| data | validation split of the Stage-1 query, one cube per GPU per step, `trainer_loop.max_steps` 32 → **256 cubes per checkpoint** (inference: 4 → 32 cubes) | AMG encodes one volume per call; 256 cubes ≈ 12 k instances is enough for a first ranking, cheap |
| image formation | recipe transforms (PSF → Poisson-Gaussian noise → Normalize), seeded | as in training |
| AMG | `points_per_side` 8 (512 clicks), `points_per_batch` 16, `pred_iou_thresh` 0.5, `stability_score_thresh` 0.7, `box_nms_thresh` 0.5, `use_m2m`, no crops, no small-region filter | model-config defaults; tuned later |
| evaluator | `sam2_instance_evaluator`: class-agnostic, GT from the label map, COCO mask/box mAP (IoU 0.50:0.95), instance mIoU + match recall (IoU 0.5), box F1, IoU-head calibration | plan §3 |
| plots | `instance_overlay`: per cube one PDF, pages every 16 z-slices, rows background / boxes (GT vs pred) / masks (GT vs pred), orthogonal views | inspection only, nothing saved as volumes |
| hardware | 1 b300 node, 8 GPUs, local launcher (`scripts/utils/run_queue.sh`), one run at a time | node-local flush |

Code (uncommitted, for review): `datasets.eval_split` (test/predict loaders on the validation split), `trainer_loop.max_steps`
honoured by TestTrainer/Inferencer, the Inferencer drops the SAM2 model-private GT view before dispatch to the CPU viz actors,
`InstanceSegmentationEvaluator` accepts the inherited `training_metrics` key.

## 3. How we evaluate

- Primary: **mask mAP** (`epoch_loss/test/mask_map_median` in `<outdir>/logs/scalars/epoch_logbook.csv`); secondary: instance
  mIoU + match recall, box mAP/F1, IoU-head MAE/Pearson (is the IoU head a usable confidence?).
- Noise floor at 256 cubes: Δ mAP ≈ 0.01–0.02 (to be checked by re-running one checkpoint on a second 256-cube slice).
- Decision: ranking only at epoch 6 (all runs are mid-cosine); the lr decision waits for the final checkpoints. AMG thresholds
  are tuned on the winner afterwards (precision/recall trade, `pred_iou_thresh`/`stability_score_thresh`).

## 4. Data

| run | ckpt | cubes | s/cube | mask mAP | mask mIoU | match recall | box mAP | box F1 | IoU-head MAE | Pearson | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `eval_lr_0p0002_ep6` | step 23076 | 128 | 5.5 (max 30) | **0.624** | 0.934 | 0.706 | 0.407 | 0.774 | 0.127 | 0.60 (Spearman 0.90) | 4,908 predictions; precision@0.5 0.99, coverage@0.5 1.00 |
| `eval_lr_0p0004_ep6` | step 23076 | 128 | 5.5 | **0.625** | 0.939 | 0.704 | 0.417 | 0.772 | 0.136 | 0.58 (Spearman 0.89) | 5,139 predictions; precision@0.5 0.99 |
| `eval_lr_0p0001_ep6` | step 23076 | 128 | 5.5 | **0.624** | 0.931 | 0.706 | 0.412 | 0.778 | 0.115 | 0.60 (Spearman 0.90) | 4,818 predictions; precision@0.5 0.99 |

Attempts:

| # | config | outcome | notes |
|---|---|---|---|
| 1 | `eval_lr_0p0002` | **died at the first step** (13:41) | `mat1 and mat2 must have the same dtype (BFloat16 vs Float)` in the patch embedding: the eval trainers run the raw fp32 module on the preprocessor's bf16 input, while training let FSDP cast parameters per forward. Fix: the test/predict loops run under `torch.autocast` at `parallelism.training.mixed_precision_param` (`_eval_autocast`, loops.py). Sandbox extraction + Ray start + model build ≈ 6 min before the first step. |
| 2 | `eval_lr_0p0002` | **killed 14:31** (alive but too slow) | first step ran, but workers sat at 100 % of one CPU core and 0 % GPU: `MaskData.cat` re-copied the growing full-resolution mask stack on every one of the 32 click batches (quadratic), then the evaluator's per-prediction bincount ran on the CPU. Fixes: single concatenation at the end of the batch loop (`MaskData.concat`), evaluator `compute_device: cuda`, per-step progress line in the log, 16 cubes/GPU for the flush phase. |
| 3 | `eval_lr_0p0002` | **died after the first cube** (14:40) | AMG + GPU scoring of cube 1 took < 2 min; then `indices should be either on cpu or on the same device` in the evaluator: with `compute_device: cuda` the IoU rows come back on the CPU while the score mask stayed on the GPU. Fix: scores / predicted IoUs move to the CPU once the per-voxel work is done (the streaming metrics keep CPU state for the cross-rank gather). |
| 4 | `eval_lr_0p0002` | **passed** 14:54 (16 cubes/GPU in 1.5 min, 5.5 s/cube median) | metrics in the table above; `queue5` = 4e-4, 1e-4 |
| 5 | `eval_lr_0p0004` | **passed** 15:08 | same speed |
| 6 | `eval_lr_0p0001` | killed at start (15:1x) | the node allocation (i04u02) ended; relaunched 15:20 on i01u02 (`queue6`, followed by `infer_lr_0p0002`) |
| 7 | `eval_lr_0p0001` | **passed** 15:30 | table complete: the three checkpoints are indistinguishable (mAP 0.624/0.624/0.625, recall 0.706 each) |
| 8 | `infer_lr_0p0002` | ran to completion 15:40 (4 cubes/GPU, 8–13 s/cube) but **0 of 32 plots written** | every viz handler call failed with `'numpy.ndarray' object has no attribute 'unbind'`: the overlay converts GT boxes with a tensor-only helper while the viz worker receives numpy. Fix in `save_instance_predictions` (+ test). Note: the run still exits 0 — viz failures are logged, not fatal; check `visualize_successful` / the PDF count. |
| 9 | `infer_lr_0p0002` | launched 15:42 | GT-box fix |

Reflections: (fill after the runs)
