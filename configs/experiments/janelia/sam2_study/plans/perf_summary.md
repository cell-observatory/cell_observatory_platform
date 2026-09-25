# SAM2 step-time study — summary (2026-09-03)

Setup for every row: cube 128×384×512, mm 48, 8× B300 (275 GB), torch-native FSDP2, point-sampled focal/dice, 200-step runs,
medians over steps 50–200. Baseline = today's code at 7 correction rounds, bs 2/GPU. Full details and every attempt:
`2026-09-02-sam2-stage0c-cubes.md` (matrix), `2026-09-02-sam2-step-time-debug.md` (the 14 s regression), `2026-09-02-sam2-perf-proposals.md`.

## 1. Results

| row | rounds | bs/GPU | flags | step s | GiB (%) | samples/s/GPU | vs baseline |
|---|---|---|---|---|---|---|---|
| regression (before fix) | 7 | 1 | cudnn.benchmark on | ~11–14 | 108 | 0.07 | — |
| **baseline (K2)** | 7 | 2 | `cudnn_benchmark: false` | 1.96 | 215 (78) | 1.02 | 1.0× |
| compile (ViT blocks) | 7 | 2 | | 1.93 | 213 | 1.04 | +1 % |
| encoder act-ckpt | 7 | 2 | | 1.95 | 208 | 1.02 | 0 |
| decoder act-ckpt | 7 | 2 | | 2.24 | **107** | 0.89 | −13 % (memory ×0.5) |
| decoder act-ckpt, bs 4 | 7 | 4 | | 4.29 | 213 | 0.93 | −9 % |
| MX-FP8 (all linears) | 7 | 2 | | 2.84 | 215 | 0.70 | −31 % |
| criterion act-ckpt off (P6) | 7 | 2 | | 1.76 | 215 | 1.14 | **+12 %** |
| decoder up-scaling as GEMM+shuffle (P2) | 7 | 2 | | 1.88 | 215 | 1.07 | +5 % |
| P2 + prompt down-scaling as GEMM (P3) | 7 | 2 | | 1.81 | 215 | 1.10 | +8 % |
| low-res correction loop + IoU (P1) | 7 | 2 | | 1.40 | **157** | 1.43 | **+40 %** |
| rounds 3 | 3 | 2 | | 1.12 | 133 | 1.79 | +75 % |
| rounds 3, bs 3 | 3 | 3 | | 1.55 | 199 (72) | 1.93 | +89 % |
| rounds 1 | 1 | 2 | | 0.76 | 92 | 2.63 | +158 % |
| rounds 1, bs 5 | 1 | 5 | | 1.49 | 227 (83) | **3.36** | **+230 %** |
| rounds 1, bs 5, compile | 1 | 5 | | 1.49 | 221 | 3.36 | 0 |
| rounds 1, bs 5, MX-FP8 | 1 | 5 | | 2.01 | 227 | 2.48 | −26 % (ViT 228 → 452 ms) |
| P1+P2+P3 (P123) | 7 | 2 | | 1.27 | 157 (57) | 1.58 | +55 % |
| P4 mask rows global-max | 7 | 2 | | 1.87 | 215 | 1.07 | +5 % (rank wait 0.33 s → 0.02 s) |
| P4 fixed rows + cuDNN benchmark on | 7 | 2 | | 1.89 | 215 | 1.06 | +4 % (= global-max; 2.5 min one-off tuning) |
| S1: rounds 3 + P6, bs 3 | 3 | 3 | | 1.42 | 198 (72) | 2.11 | +107 % |
| S2: rounds 3 + P1+P2+P3 + P6, bs 4 | 3 | 4 | | 1.36 | 184 (67) | 2.94 | +188 % |
| S2b: same at bs 5 | 3 | 5 | | 1.68 | 232 (84) | 2.98 | +192 % (bs 5 = memory ceiling at 3 clicks; no gain over bs 4) |
| **S3: rounds 1 + P1+P2+P3 + P6, bs 7** | 1 | 7 | | 1.60 | 209 (76) | **4.38** | **+330 %** |
| **S3 at bs 8** (14 steady steps; run hit an epoch boundary) | 1 | 8 | | 1.68 | 237 (86) | **4.77** | **+368 %**; +9 % vs bs 7 → recipe |
| S3 at bs 10 + decoder act-ckpt | 1 | 10 | ac-dec | OOM | | | decoder act-ckpt frees too little at 1 click |

Epoch time (8 GPUs) = rows / (samples/s/GPU × 8): at 3.36 samples/s/GPU the 65 k Step-1 subset takes **40 min**, the full
full in-bounds 128×384×512 set (252,431 annotated cubes on the 2026-09-03 snapshot; the fringe fix drops the overflowing cubes) **2.6 h**; at the 7-round baseline those were 2.2 h / 8.6 h.

## 2. Where the time goes (best setup: 1 click, full stack, bs 7 → 1.60 s/step, 56 volumes)

| phase | ms | share | notes |
|---|---|---|---|
| ViT-L backbone (6144 tokens × 7) | 286 | 18 % | the only GEMM-bound part |
| decoder passes (2, GEMM+shuffle upscaling) | 311 | 19 % | two-way transformer + upscaling over K·V/8; still the largest single item |
| error-point sampling (stride 4) | 98 | 6 % | was 137 at full res |
| loss (low-res IoU, no ckpt recompute) | 91 | 6 % | was 143 |
| GT mask blocks + targets | 27 | 2 % | |
| other forward (prompt prep, mask-prompt downscale, feature gather to K rows) | ~206 | 13 % | grows with bs; next candidate to look at |
| backward | 655 | 41 % | |
| optimizer, sync, cross-rank wait | ~15 | 1 % | rank wait max 12 ms |

Reading: after the cuts the backbone is ~18 % and the decoder passes ~19 %; the remaining 60 % is still per-mask work
(decoder upscaling, sampling, loss, "other forward") and its backward, i.e. memory-bandwidth-bound traffic over K·V/64 and
K·V/8 tensors rather than tensor-core math. Compile and FP8 were re-checked at 1 click / bs 5 and still give nothing
(FP8 +35 % slower), so the ViT is not the place to push.

## 3. Easy changes worth considering (config or flag only)

| change | how | effect measured / expected | caveat |
|---|---|---|---|
| **fewer correction rounds** | `num_correction_pt_per_frame: 3` (or 1) | 1.96 → 1.12 s (3) / 0.76 s (1) at bs 2 | recipe: fewer simulated clicks per step; check mAP in Step 2 |
| **criterion act-ckpt off** | `criterion_args.activation_checkpoint: false` | −10 % step, no memory cost at bs 2 | none |
| **low-res correction loop** | `correction_on_low_res: true` + `criterion_args.iou_on_low_res: true` | −29 % step, −27 % memory at 7 rounds | clicks quantized to stride-4 cells; IoU target at stride 4 |
| **GEMM+shuffle up/down-scaling** | `mask_decoder_args.upscaling: linear_shuffle`, `prompt_encoder_args.mask_downscaling: linear_shuffle` | −7 % | exact; old checkpoints need the state-dict converter |
| **fill VRAM with batch after the cuts** | bs 8 at 1 click with the full stack (237 GiB, 86 %); bs 5 at 3 clicks | +42 % samples/s over bs 5 without the stack; bs 10 with decoder act-ckpt OOMs | 86 % leaves little headroom: watch `num_alloc_retries` in long runs |
| equal mask rows (P4) | `datasets.preprocessor.mask_rows: global_max` | −5 % at bs 2 (rank wait 0.33 → 0.02 s); ~0 at bs ≥ 5 | statistically equivalent, not bit-identical |
| keep off | compile, encoder act-ckpt, MX-FP8, decoder act-ckpt (unless a shape must fit) | no gain / slower | — |

Not easy (needs real work, listed for completeness): balancing instance counts across ranks in the loader; removing the
remaining full-res GT mask block; a fused decoder upscale+hypernetwork kernel; a different interactive protocol.

## 4. Recipes (the deliverable)

| file | clicks | bs/GPU | measured | epoch on 65 k / 252 k |
|---|---|---|---|---|
| `configs/experiments/janelia/tests/2026_09_02/sam2/recipe_r1.yaml` | 1 | 8 | 1.68 s/step, 237 GiB (86 %), 4.77 samples/s/GPU | 29 min / 1.8 h |
| `configs/experiments/janelia/tests/2026_09_02/sam2/recipe_r3.yaml` | 3 | 5 | 1.68 s/step, 232 GiB, 2.98 samples/s/GPU | 46 min / 2.9 h |

Both set: `correction_on_low_res` + `iou_on_low_res`, `upscaling: linear_shuffle`, `mask_downscaling: linear_shuffle`,
`criterion_args.activation_checkpoint: false`, `mask_rows: global_max`, `cudnn_benchmark: false`, mm 48, `in_bounds_only: true`.
They inherit the study base (Janelia paths, torch-native FSDP2, point loss, lr 1e-4 placeholder) and are meant to be inherited
by the Step-1 configs (`2026-09-03-sam2-stage1-lr-plan.md`). Not yet gated on the recipe: the DCP resume check
(`stage0c/C1_resume_a/b`, configs ready) — run it once before the first chained job.

## 5. Fixed on the way (uncommitted, for review)

`cudnn_benchmark: false` in the study base (the 14 s regression); argmax-key point sampler; per-phase CUDA-event timers
(`training/phase_timer.py`); MX-FP8 contiguity + `include_fqns`; sandbox extract-once + stale-lock cleanup in
`cluster/ray_local_cluster.sh`; P1–P4 behind flags with tests; LSF job chaining (`clusters.chain_jobs`, `cluster/chain_lib.sh`, `TRAINING_DONE` marker, tests); `scripts/utils/{run_queue.sh,bench_table.py,trace_*.py,nccl_check.py}`.
