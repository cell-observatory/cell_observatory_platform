# Evaluation + inference configs (SAM2 study)

Two overlays turn any run's training config into an eval-time job on that run's own held-out validation cubes:

| overlay | job | what it does | output |
|---|---|---|---|
| `test_overlay.yaml` | `test` (TestTrainer) | unprompted automatic mask generation → instance evaluator (mask/box mAP, mIoU + match recall, box F1, IoU-head calibration), 16 cubes/GPU | `$DATA_DIR/sam2_study/eval/<name>/logs/scalars/epoch_logbook.csv`; table: `python scripts/utils/eval_table.py $DATA_DIR/sam2_study/eval` |
| `predict_overlay.yaml` | `predict` (Inferencer) | same masks, plots only: one PDF per cube (GT vs pred boxes and masks, 3 orthogonal views, every 16 z), 4 cubes/GPU | `$DATA_DIR/sam2_study/inference/<name>/visualizations/*.pdf` |

A leaf is three lines (see `stage1p5/`, `stage2/`, `stage3/`):

```yaml
# @package _global_
defaults:
  - /experiments/janelia/sam2_study/<the run's training config>     # data query, shape, channels, model variant
  - /experiments/janelia/sam2_study/eval/test_overlay               # or predict_overlay
  - _self_
experiment_name: eval_<run>_<epoch>
paths:
  pretrained_checkpointdir: <dir holding step-N/>                   # copy the step dir out of a live run first (keep_latest_k 3)
```

Rules: the run's config must be the FIRST default (same query + `seed` + `split` ⇒ identical validation rows via
`datasets.eval_split: val`); batch is one cube per GPU (AMG encodes one volume per call); launch on a b300 node in local
mode, one at a time: `bash scripts/utils/run_queue.sh $DATA_DIR/sam2_study/eval/logs experiments/janelia/sam2_study/eval/<dir> <leaf>`.
A cube takes ~5 s (eval) / ~10 s (inference incl. the PDF); start-up ≈ 6 min. Viz failures are logged, not fatal:
check the PDF count. Recall probe: `eval_lr_0p0002_pps16.yaml` (4,096 clicks per cube instead of 512).

| study stage | run configs | eval leaves |
|---|---|---|
| Stage 1 lr sweep | `stage1_lr/lr_base` (+ multi_run `lr_sweep`) | `eval_lr_0p000{2,4,1}.yaml`, `infer_lr_0p000{2,4,1}.yaml` (epoch-6 checkpoints copied to `$DATA_DIR/sam2_study/eval/ckpts/`) |
| Stage 1.5 tile adaptation | `stage1p5_tile/tile_adapt` (to write; `sweep/sam2_c1024_bs4` is the shape/bs stand-in) | `stage1p5/eval_lr_0p0002_ep6_zeroshot_c1024` (512-trained checkpoint at 1024, runnable now), `eval_tile_adapt`, tile-path `eval_tile_{crop,resize}1024` (whole tiles, cropped/resized to 1024 — X-sliding windows are not implemented) |
| Stage 2 channels / dropout | `sweep/sam2_ch_{attnpool,concat,dropout,membrane}_*` as stand-ins for the ablation configs | `stage2/eval_ch_*` |
| Stage 3 scale (ViT-H, epochs, init) | `sweep/sam2_vith_c512_bs7` stand-in | `stage3/eval_vith` (epochs/init variants reuse `eval_lr_*` with another checkpoint dir) |

Held-out *tiles* (plan D3, tile-disjoint): not defined; every tile is in the training query. Until then the validation
split (unseen cubes, seen tiles) is the held-out set.
