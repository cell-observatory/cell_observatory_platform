# Stage 2 — channel parametrisation, channel dropout, interaction protocol

2026-09-09 · status: **configs written, gate minis launched** · configs: `stage2_ablations/` · base: `recipe/recipe_r1` + lr 2e-4, 6 epochs

## 1. What we are investigating and why

Which way of feeding the 5 channels to the ViT, whether dropping channels at train time helps robustness, and how many simulated
clicks the training protocol needs. Now because the LR sweep is settled (`stage1_lr_plan.md`: 2e-4 / 4e-4 / 1e-4 all reach val
0.40–0.41 by epoch 7; Hugo picked 2e-4 for its faster early convergence) and the sweep plateaued by epoch ~6, so ablations get a
**6-epoch** schedule instead of 20. Evidence for the variants: setup sweep (`setup_sweep.md`): attn-pool costs the same as the
joint embed, concat 5.4× (deferred), dropout free, 3 clicks = bs 5/GPU.

## 2. Runs

Variable per run (one change each vs the baseline); one B300 node per run, chains of 4 h links:

| run | change | bs/GPU |
|---|---|---|
| `abl_baseline` | none: joint embed, 1 click, gt-prob 0.1 | 8 |
| `abl_A1_attnpool_factorized` | per-channel tokens + factorized (localization × fluorophore) embedding, attn-pool fusion | 8 |
| `abl_A1_attnpool_none` | per-channel tokens, no channel identity | 8 |
| `abl_A3_membrane_only` | C = 1 (membrane) | 8 |
| `abl_A3_cytosol_only` | C = 4 (cytosol) | 8 |
| `abl_B_dropout_0p25_shuffle` | ChannelDropout p 0.25 + shuffle on A1-factorized | 8 |
| `abl_B_dropout_0p5_shuffle` | ChannelDropout p 0.5 + shuffle on A1-factorized | 8 |
| `abl_D_clicks0` | 0 correction clicks | 8 |
| `abl_D_clicks3` | 3 correction clicks | 5 |
| `abl_D_gtprob0p0` | correction click never from GT | 8 |
| `abl_D_gtprob0p3` | correction click from GT with p 0.3 | 8 |

Deferred: concat fusion (5.4× cost, only if A1 wins), no-shuffle dropout variants, box prompts (needs the collator box format).

Held constant: recipe_r1 (1 click, low-res click loop, GEMM up/down-scaling, criterion ckpt off, mm 48, uniform eval sampling,
PSF + sensor noise, 2 excluded tiles), lr 2e-4, wd 2e-5, warm-up 1 epoch, cosine to 5 %, **6 epochs**, split 0.02, seed 42,
full 128×384×512 set (30,768 rows per rank), 55-min + per-epoch checkpoints, `chain_jobs: 6`.

Budget: 6 epochs × 2.4 h = 14.5 h per run (4 links) × 11 runs ≈ 160 node-hours. Gate minis (`mini_channel`, `mini_clicks3`,
`mini_clicks0`, `mini_membrane`; 1024 rows, 2 epochs, 15-min links) run first, one per new code path.

## 3. How we evaluate

Val loss (total and dice) per epoch, decision at epoch 6 vs the baseline; Δ = 0.01–0.02 is noise (the LR sweep's spread at
convergence). Held-out mAP (`eval/test_heldout`, being written) on the epoch-6 checkpoints for the top candidates. Rules: A
winner = lowest val at 6 (attn-pool must beat joint by > Δ to justify the vocab machinery); B is judged on top of A1; D: fewer clicks
win if within Δ (they are cheaper: 0 clicks ≈ 0.76 s/step vs 1.68).

## 4. Data

| run | epochs | val total @6 | val dice @6 | train @6 | wall-clock | notes |
|---|---|---|---|---|---|---|
| (per epoch: `python scripts/utils/epoch_table.py --runs $DATA_DIR/sam2_study/stage2_ablations/abl_* --md`) | | | | | | |
| `S2_baseline` | 1 | 0.801 | 0.467 | 1.49 | 2.26 h/epoch | val iou 0.130 |
| `A3_membrane_only` | 1 | 0.740 | 0.440 | 1.35 | 1.93 h/epoch | val iou 0.125 |
| `A3_cytosol_only` | 1 | 1.098 | 0.683 | 1.78 | 2.17 h/epoch | val iou 0.175; no membrane = large loss |
| `B_dropout_0p5_shuffle` | 1 | 0.739 | 0.446 | 1.37 | 2.32 h/epoch | val iou 0.127; attn_pool + factorized, same cost as joint |
| `D_gtprob0p0` | 1 | 0.821 | 0.498 | 1.50 | 2.27 h/epoch | val iou 0.148 |
| `D_gtprob0p3` | 1 | 0.823 | 0.466 | 1.52 | 2.27 h/epoch | val iou 0.149; 0.0 / 0.1 / 0.3 within noise at epoch 1 |
| `D_clicks0` | 1 | 0.510 | 0.300 | 0.88 | 1.80 h/epoch | val iou 0.094; NOT comparable: the loss is a SUM over prediction rounds (1 here, 2 for 1 click, 4 for 3 clicks) -> mAP decides D |
| `D_clicks3` | 1 | 1.163 | 0.712 | 2.14 | 3.41 h/epoch | val iou 0.208; 4 rounds summed = 0.29/round vs 0.40 (1 click) and 0.51 (0 clicks); bs 40, 1.45 s/step -> 6 epochs = 20 h |

Launch log (2026-09-09, tag `runs/sam2-stage2-2026-09-09`, one LSF chain per leaf, outdirs `$DATA_DIR/sam2_study/stage2_ablations/abl_*`):

| when | what | fix |
|---|---|---|
| 16:35 | `mini_clicks0`: `sam_outputs` unbound with 0 correction clicks | 9721380: loop skipped -> prompt-only prediction is final |
| 16:57 | `A1_attnpool_factorized`, `B_dropout_0p25_shuffle` link 1: channel_vocab refused the empty `<outdir>/checkpoints` resume dir | 49ae779: empty resume dir = fresh start; relaunched |
| 17:00 | all 11 chains submitted (10 + `D_clicks0` after its smoke); 8 nodes -> ~half queued, links interleave | – |

Reflections: (fill)
