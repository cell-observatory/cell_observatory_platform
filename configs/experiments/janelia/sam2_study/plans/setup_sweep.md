# Run types and the setup sweep (bs / levers / bug-flush per run type)

2026-09-03 · status: **sweep running; moved to i07u02 (LSF 153968320) at 20:38 after the first allocation ended; local mode, one run at a time** · configs: `configs/experiments/janelia/tests/2026_09_03/sweep/` (+ `pretrain/`) · rows: `python scripts/utils/bench_table.py $DATA_DIR/sam2_study/sweep` · logs: `$DATA_DIR/sam2_study/sweep/logs/`

## 1. What we are investigating and why

Every run type the study needs (plan_sam2_v2 stages 1–3 incl. the new channel-embedding code, the Stage-1.5 tile adaptation,
and the MAE/JEPA pretraining tranche) must have a **known-good setup before it is launched unattended on LSF**: the bs/GPU
that fills ~80 % of the 275 GB, the levers that help (compile, act-ckpt, FP8 — measured, not assumed), its samples/s, and
zero config/code bugs. Only the single-scale SAM2 recipe at 128×384×512 has that today (perf summary: bs 8, 1.68 s/step,
4.77 samples/s/GPU). Each probe = 200 steps, medians over steps ≥ 50, ~4 min cluster start + 200 × step; one high bs
guess per type, one back-off on OOM, one step-up if < 70 % VRAM (stage-0 rule: every run costs money).

## 2. The run types

| id | family | variant | backbone | shape | needed by | setup status | probe config |
|---|---|---|---|---|---|---|---|
| A1 | SAM2 | single-scale, 1 click (recipe_r1) | ViT-L | 128×384×512 | Stage 1 lr, Stage 2, Stage 3 | **done**: bs 8, 1.68 s, 237 GiB, 4.77 samples/s/GPU | `2026_09_02/sam2/recipe_r1` |
| A1b | SAM2 | single-scale, 3 clicks (recipe_r3) | ViT-L | 128×384×512 | Stage 2 D (interaction) | **done**: bs 5, 1.68 s, 232 GiB, 2.98 | `recipe_r3` |
| A2 | SAM2 | multi-scale (Hiera-L + high-res features) | Hiera-L | 128×384×512 | — | **dropped from the sweep** (Hugo 20:25); geometry fix in `sam2_ms_base` runs past the decoder, bs 8 OOM | `sweep/sam2_ms_c512_bs{8,6}` |
| A3a | SAM2 | channel_adaptive **attn_pool** × embed {none, localization, fluorophore, factorized} (same tokens/compute; one probe covers all four) | ViT-L | 128×384×512 | Stage 2 A1 | **done**: bs 8, 1.67 s, 239 GiB = joint | `sweep/sam2_ch_attnpool_c512_bs8` |
| A3b | SAM2 | channel_adaptive **concat** (5× tokens) | ViT-L | 128×384×512 | Stage 2 A2 | **done**: bs 2 = 2.26 s, 114 GiB, 0.88 samples/s/GPU (bs 4 predicted 83 %) | `sweep/sam2_ch_concat_c512_bs2` |
| A3c | SAM2 | channel dropout p 0.25 + shuffle (on A3a) | ViT-L | 128×384×512 | Stage 2 B | **done**: bs 8, 1.65 s, 237 GiB = A3a | `sweep/sam2_ch_dropout_c512_bs8` |
| A3d | SAM2 | membrane-only C=1 / cytosol-only C=4 (joint embed) | ViT-L | 128×384×512 | Stage 2 A3 | **done** (membrane): bs 8, 1.61 s, 229 GiB; cytosol-only = same recipe | `sweep/sam2_ch_membrane_c512_bs8` |
| A3e | SAM2 | `<unk>` token training p 0.2 (on A3a) | ViT-L | 128×384×512 | Stage 2 A1 robustness | no perf change; config only (`unk_token_p`) | — |
| A4 | SAM2 | recipe at **128×384×1024** (Stage 1.5) | ViT-L | 128×384×1024 | Stage 1.5 tile adaptation | **done**: bs 4, 2.00 s, 237 GiB, 2.0 samples/s/GPU | `sweep/sam2_c1024_bs4` |
| A4t/A4r | SAM2 | tile-mode **crop** / **resize** to 128×384×1024 on the recipe (`sample_type: tile`) | ViT-L | tile → 1024 | Stage 1.5 alternatives; inference path | **done**: bs 2, mm 96: crop 1.74 s / 214 GiB, resize 1.83 s / 215 GiB (≈1.1 samples/s/GPU, no I/O wait) | `sweep/sam2_tile_{crop,resize}1024_bs2` |
| A10 | SAM2 | **image formation on synthetic data** (sanity PNGs done: `$DATA_DIR/sam2_study/psf_noise_check/`; raw intensities are count-like, cytosol 0–6.5 k, membrane ≤ 18 k, no clipping after the sensor model): PSF convolution + mixed Poisson-Gaussian noise (`ConvolveWithPSF` → `MixedPoissonGaussianNoise` → `Normalize`, the denoising-preprocessor order) | ViT-L | 128×384×512 | **every real training run** (Stage 1 lr onwards; Hugo 21:10) | **done**, transforms now in `recipe_r1`/`recipe_r3` (Stage-1 lr configs inherit them); PSF = **measured `CamA_Arc_OMW.tif` since 2026-09-08** (Hugo's copy in hph/data; centred, uncompressed copy under `$DATA_DIR/psf/`; 75×52×286 voxels, 33 % of the energy within ±5 voxels, 55 % within ±10 — long tails; pixel size assumed 0.1 µm isotropic, no tags in the file). Earlier probe used a Gaussian stand-in, sensor params from `tasks/denoising.yaml`; sanity PNGs → `$DATA_DIR/sam2_study/psf_noise_check/` | `sweep/sam2_psf_noise_c512_bs8` |
| A5 | SAM2 | **ViT-H** | ViT-H | 128×384×512 | Stage 3 S3 | **done**: bs 7 = 1.90 s, 238 GiB (87 %), 3.68 samples/s/GPU (bs 5: 1.39 s, 62 %, 3.60) | `sweep/sam2_vith_c512_bs{5,7}` |
| A6 | SAM2 | pretrained init (MAE/JEPA → SAM2, key map, freeze variants) | ViT-L | 128×384×512 | Stage 3 S2 | same perf as A1; needs a pretrained checkpoint + `pretrained_key_map` smoke | — |
| A7 | SAM2 | DeepSpeed ZeRO-2 twin | ViT-L | 128×384×512 | plan D1 (once) | deferred (`stage0c/D1_ds_c512`) | — |
| A8 | SAM2 | eval job (`job_type: test`, held-out mAP; tile-size eval for Stage 1.5) | any | 512 / tile | every stage's decision | **config missing** (`eval/test_heldout`), `pt_sampling_for_eval` guard | — |
| A9 | SAM2 | DCP resume gate (save@20 → resume 20→40) on the recipe | ViT-L | 128×384×512 | first chained job | configs ready (`stage0c/C1_resume_a/b`), not run | — |
| B1 | pretrain | **MAE** ViT single-scale, random 0.75 | ViT-L (then B, H, g) | 128×384×512×5 | pretraining tranche, Stage 3 S2 init | tonight (L) | `sweep/pretrain/mae_vit_bs64` |
| B2 | pretrain | **JEPA** ViT single-scale, blocked 0.7–0.9 × 2 | ViT-L (then B, H, g) | 128×384×512×5 | same | tonight (L) | `sweep/pretrain/jepa_vit_bs96` |
| B3 | pretrain | **MAE Hiera-L** (plain Hiera, hiera_mu masking, no DA) | Hiera-L | 128×384×512×5 | hierarchical vs fixed-scale (the only multi-scale variant wanted) | tonight | `sweep/pretrain/mae_hiera_bs16` |
| B4 | pretrain | JEPA Hiera | Hiera-L | — | — | **dropped** (Hugo: Hiera multi-scale only for MAE, plain, no DA / special kernels) | — |
| B5 | pretrain | masking ablations: random vs block, ratio {0.6, 0.75, 0.9}, block aspect, #targets | ViT-L | same | masking study | perf changes only through visible-token count: bench the **lowest** ratio corner (most tokens) once B1/B2 land | later |
| B6 | pretrain | model × data scale: {B, L, H, g} × {100 k, 500 k, 1 M, 2 M} cubes | B/L/H/g | same | scaling study | bs per size: B and H/g one probe each after L; data size is a `max_rows` knob. Live counts: 128×384×512 has 344 k cubes (252 k annotated + 91 k not), 128×256×256 has 1.42 M → the 1 M / 2 M points need the smaller shape (`2026-09-03-pretraining-plan.md`) | later |
| B7 | pretrain | DINO / iBOT | ViT-L | same | SSL-objective comparison | **not runnable**: `optimizations/models/dino` has no module lists (TODO), multisequence preprocessor commented out | blocked |
| B8 | pretrain | deformable-attention variants (`*_hiera_da`) | Hiera-L | — | — | **not wanted** (Hugo: no fancy kernels) | — |
| L | levers | compile / act-ckpt / MX-FP8 / bs step-up on each winner | — | — | — | SAM2: all measured (off). Pretraining is GEMM-bound → **measure again** on B1/B2 (FP8 + compile may pay here) | after B1/B2 |

Fixed for every probe: torch-native FSDP2, bf16, `cudnn_benchmark: false`, 8× B300, local launcher, `max_rows` ≥ 200 × bs
(step-mode runs must not cross an epoch boundary), 2026-09-03 DB snapshot, `in_bounds_only`.

## 3. Decision rules

- **bs**: largest that runs 200 steps with `num_alloc_retries = 0` and max_reserved ≤ ~85 %; if two bs give the same
  samples/s, the smaller one (headroom for long runs).
- **lever**: on only if ≥ 8 % more samples/s at the same bs, or it lets a bigger bs fit and that bs is ≥ 8 % faster.
- **bug**: fix in the working tree (uncommitted), rerun once as `_r2`; a design decision goes to Hugo via Slack.
- Rates feed the budget tables of the stage docs (lr plan, Stage 1.5 plan, pretraining plan).

## 4. Data

| run | family | bs/GPU | steps | step s | GiB (%) | samples/s/GPU | wait ms | retries/OOM | outcome | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `sam2_c1024_bs4` | A4 | 4 | 200 | 2.00 | 237 (86) | 2.00 | 9 | 0 / 0 | **pass → BS_1024 = 4** | 16 samples/s/node; 101 Mvox/s/GPU (c512 recipe: 120); fwd/bwd 1038/949 ms; no headroom for bs 5 |
| `sam2_ms_c512_bs8` r1 | A2 | 8 | 0 | | | | | | **crash** (at 128³, pre-G1) | `sam_head.py:369 predict_masks`: dense PE grid (stride 16) vs decoder src (coarsest Hiera level, stride 64) — patch 16 was fed to Hiera; fix = ABC multiscale geometry (data patch 4, SAM-side patch 16), r2 queued (queue14) |
| `sam2_ms_c512_bs8` r2 | A2 | 8 | 0 | | OOM | | | 0 / 8 | **OOM** (geometry fix works: past the decoder) | 264 GiB allocated + 13.5 GiB request at step 0 → ~35 GiB/sample (Hiera patch 4: 393 k stage-1 tokens) |
| `sam2_ms_c512_bs6` | A2 | 6 | | | | | | | **skipped** (Hugo 20:25: multi-scale only as plain Hiera for MAE; drop from tonight) | leaf kept for later |
| `sam2_ch_attnpool_c512_bs8` | A3a | 8 | 200 | 1.67 | 239 (87) | 4.79 | | 0 / 0 | **pass → same recipe as joint (bs 8)** | = joint embed (1.68 s, 237 GiB); covers none/localization/fluorophore/factorized |
| `mae_vit_bs64` r1 | B1 | 64 | 0 | | | | | | **killed** (G4) | host buffer slots |
| `mae_vit_bs64` r2 | B1 | 64 | 0 | | | | | | **crash** (G6: 6-channel buffer reaches patchify) | G4 fixed (no hostRegister failures, workers alive) |
| `mae_vit_bs64` r3 | B1 | 64 | | | | | | | running (queue22, 22:08; Hugo 22:05: run all pretraining probes + sizes/masking/levers) | |
| `jepa_vit_bs96` r1 | B2 | 96 | 0 | | | | | | **killed** (G4, 8/8 ranks `hostRegister failed`) | confirms G4 |
| `jepa_vit_bs96` r3 | B2 | 96 | | | | | | | queued (queue22) | G4 + G6 fixes in |
| `sam2_ch_concat_c512_bs2` | A3b | 2 | 200 | 2.26 | 114 (41) | 0.88 | | 0 / 0 | **pass; 5.4× the cost of joint per sample** | VRAM law → bs 4 ≈ 228 GiB (83 %) = the setting if A2 is ever run; not benched (economy) |
| `mae_ch_attnpool` | B1+A3a | 64 | | | | | | | queued (queue22) | channel-adaptive pretraining (channel_ids now emitted by `RayPreprocessor`) |
| `jepa_ch_attnpool` | B2+A3a | 96 | | | | | | | queued (queue22) | |
| `mae_vit_B_bs128` | B6 | 128 | | | | | | | queued (queue22) | ViT-B size point |
| `mae_vit_H_bs32` | B6 | 32 | | | | | | | queued (queue22) | ViT-H size point |
| `jepa_vit_H_bs48` | B6 | 48 | | | | | | | queued (queue22) | |
| `mae_vit_mask0p6_bs64` | B5 | 64 | | | | | | | queued (queue22) | masking memory corner: ratio 0.6 (40 % visible) |
| `jepa_vit_smallblocks_bs96` | B5 | 96 | | | | | | | queued (queue22) | masking memory corner: target blocks 0.3–0.5 (largest context) |
| `mae_vit_compile_bs64` | L | 64 | | | | | | | queued (queue22) | torch.compile on MAE blocks |
| `mae_vit_fp8_bs64` | L | 64 | | | | | | | queued (queue22) | MX-FP8 linears |
| `jepa_vit_compile_bs96` | L | 96 | | | | | | | queued (queue22) | compile under variable context length (recompiles expected) |
| `sam2_vith_c512_bs5` | A5 | 5 | 200 | 1.39 | 171 (62) | 3.60 | | 0 / 0 | pass; under 80 % → step-up | ViT-H costs +25 % per sample vs ViT-L bs 8 |
| `sam2_vith_c512_bs7` | A5 | 7 | 200 | 1.90 | 238 (87) | 3.68 | | 0 / 0 | **pass → ViT-H recipe bs 7** (fills VRAM; only +2 % samples/s over bs 5, so bs 5–6 is as good with headroom) | ViT-H costs 1.3× ViT-L per sample (3.68 vs 4.77) |
| `sam2_tile_crop1024_bs2` | A4t | 2 | 100 | 1.74 | 214 (78) | 1.15 | ~20 | 0 / 0 | **pass; NOT I/O-bound** | tile path, random crop 128×384×1024, mm 96: 0.87 s/sample vs 0.50 on DB cubes (mm 48); bs 3 would be ~100 % → bs 2 |
| `sam2_tile_resize1024_bs2` | A4r | 2 | 100 | 1.83 | 215 (78) | 1.09 | ~20 | 0 / 0 | **pass; not I/O-bound** | whole tile resized to 128×384×1024 (X squashed ~3×), mm 96: 0.92 s/sample; = crop within 5 % |
| `sam2_psf_noise_c512_bs8` r1 | A10 | 8 | 0 | | | | | | crash (G7: PSF path not bound into the container) | |
| `sam2_psf_noise_c512_bs8` r2 | A10 | 8 | 200 | 1.57 | 235 (85) | 5.10 | ~20 | 0 / 0 | **pass → in both recipes** (Gaussian stand-in PSF) |
| `sam2_realpsf_trial` (2026-09-08) | A10 | 8 | 40 | 1.53 | 235 (85) | 5.2 | | 0 / 0 | **pass**: measured PSF `CamA_Arc_OMW` + noise on the recipe; total loss 8.3 → 2.3 over 40 steps (mask 0.27 → 0.02, IoU 0.96 → 0.00, dice ~2.0) | PNGs `$DATA_DIR/sam2_study/psf_noise_check_realpsf/` | PSF convolution + sensor noise cost nothing measurable (recipe alone: 1.68 s); `lr_base`, `chain_smoke`, `recipe_r3` and the sweep leaves now compose with the three transforms |
| `mae_hiera_bs16` r2 | B3 | 16 | | | | | | | queued (queue22) | plain Hiera MAE, patch 4³ |
| `jepa_hiera_bs24` | B4 | 24 | | | | | | | **skipped** (Hugo 20:25) | config renamed `.skipped` |
| `sam2_ch_dropout_c512_bs8` | A3c | 8 | 200 | 1.65 | 237 (86) | 4.85 | | 0 / 0 | **pass** | channel dropout p 0.25 + shuffle on attn_pool: free |
| `sam2_ch_membrane_c512_bs8` r2 | A3d | 8 | 200 | 1.61 | 229 (83) | 4.97 | | 0 / 0 | **pass** | membrane-only (C=1): −4 % step, −8 GiB vs 5 channels (patch-embed input is the only difference); r1 died with the i01u02 allocation |

Bugs / gaps found (fill as they appear):

| # | run | symptom | cause | fix | status |
|---|---|---|---|---|---|
| G1 | every config inheriting `recipe_r1`/`recipe_r3` (Stage-1 `lr_base`, `chain_smoke`, the c512 sweep leaves) | composes to `train_shape [1,128,128,128,5]` = 128³ cubes | the recipe never set the shape; only the stage-0c leaves did | `train_shape`/`input_shape` 128×384×512 added to both recipes (2026-09-03 19:12); `sam2_ms_c512_bs8_r1` ran at 128³ before the fix (result = crash test only) | fixed |
| G2 | `mae_hiera_bs32`, `jepa_hiera_bs32` (not run) | would crash on batch 1: generator wants rank-4 mask units under ZYXC, model rank-3; MAE-Hiera head needs patch masks | review of the mask generator / Hiera predictor | leaves rewritten as `mae_hiera_bs16` (patch 4³, units `[1,8,8,8]`, patch masks on) and `jepa_hiera_bs24` (patch 8³, `skip_patch_mask_generation`) after the legacy Janelia Hiera tests | fixed (untested) |
| G4 | `mae_vit_bs64` r1 (and `jepa_vit_bs96` r1) | worker killed at the first batch, no traceback (`ActorUnavailableError`), 4 ranks log `hostRegister failed` | a host buffer slot is one whole batch (`buffers.py: slot_bytes = prod(batch_shape)`); 16 slots × 64 × 300 MB = 300 GB /dev/shm per rank, 2.5 TB on 8 ranks > the 2 TB tmpfs | pretraining bases: `buffer_capacity: 2`, `max_steps: 100`, `max_rows: 81920`, `timepoint_list: null` (t=0 alone has 25 k rows of this shape) | fixed; reruns queued (queue15/16) |
| G5 | Stage-1 lr chains (other session, 16:40) | run aborts on one corrupt zarr chunk (`Invalid blosc-compressed data`, tile `20250311_mem_histone/fish1_24hpf/roi4/000x_001y_000z.zarr`) | corrupt shard in the forsynthetic mirror (46 MB; the `/synthetic` copy of that chunk is 3.7 MB and differs) | (a) `scripts/utils/scan_zarr_chunks.py` over all 4,586 training tiles running on the node (`$DATA_DIR/sam2_study/zarr_scan/`); (b) loader `datasets.read_failure_policy: zero_fill` (default) — a failed per-sample region read is zero-filled (image + mask come from one read) with one warning per region and a counter (`data/datasets/pretrain_dataset_ray.py`, `tests/data/test_loader_read_failure.py`); applied to main 20:20 | scan restarted 20:50 on i07u02 (Hugo: keep it, ~75 min; `scan_zarr_chunks.py` now reads on the shard grid, `--region shard`, ~1 tile/s; partial runs found 0 bad regions in the first 100 tiles); guard applied |
| G6 | `mae_vit_bs64` r2 (patched, G4 fixed) | `ValueError: patchify expects channels-last (C=5 ...) got (64,128,384,512,6)` at step 0 | DB cube arrays carry the instance-mask channel; `RayPreprocessor` (MAE/JEPA) does not strip object-role channels the way `SAM2VideoPreprocessor`'s prefix split does | signal-prefix split in `RayPreprocessor` before transforms (`signal_prefix_len` shared with the SAM2 split; `tests/models/test_preprocessor_ray_object_tail.py`, 6 tests) — applied to main 20:35 | fixed; MAE/JEPA probes rerun later |
| G7 | `sam2_psf_noise_c512_bs8` r1 | `FileNotFoundError` for the PSF tif inside the workers | `cluster/ray_local_cluster.sh` binds only `$DATA_DIR`, `CellObservatoryData`, the sif, the sandbox and the repo into the container — not `/groups/betzig/betziglab/hph/data` | PSFs copied to `$DATA_DIR/psf/`, leaf uses `${paths.data_path}/psf/...`; r2 queued | fixed |
| G3 | pretraining + channel-adaptive embedding | `patch_embed_type` not plumbed through MAE/JEPA meta-archs | only `MaskedEncoder` knows the switch | model side applied 19:25 (`patch_embed_type`/`patch_embed_args` in `meta_arch/{mae,jepa}/*.yaml`, `tests/models/test_pretrain_channel_adaptive.py`); `RayPreprocessor` now emits `channel_ids` (shared `_ChannelIdsMixin`), vocab node resolved per model (`channel_vocab.encoder_node_path`); applied to main 19:50, 8 more tests (`tests/models/test_preprocessor_ray_channel_ids.py`, `tests/data/test_channel_vocab.py`) | applied; probes `pretrain/{mae,jepa}_ch_attnpool` queued (queue15) |

Perf findings from the read-only reviews (to land as P-rows once implemented in the worktree; all exact-math):

| # | path | finding | expected | status |
|---|---|---|---|---|
| PP1 | MAE/JEPA encoder | patch-embed Linear runs on all 6144 tokens, masks applied after | −10–14 % MAE FLOPs, −B·240 MB | **applied to main 19:25** (uncommitted; `tests/models/test_perf_pretrain_exact.py`) |
| PP2 | MAE head | pixel head on all tokens then gather; index built with `.repeat` (1.6 GB int64) | −16 GB at bs 64 | applied 19:25 |
| PP3 | MAE/JEPA loss | two full `[B,4608,20480]` temporaries | −24 GB at bs 64 | applied 19:25 |
| PP4 | JEPA | both encoders return the unused full `patches` copy | −24 GB at bs 96 | applied 19:25 |
| PP5 | SAM2 decoder (production recipe too) | dense PE `repeat_interleave` + fp32 recompute per pass: ~5 GB transient | headroom at 86 % | applied 19:25 (`get_dense_pe` cache + `expand`) |
| PP6 | JEPA masking | `randint` without the seeded generator → ranks diverge; ctx/tgt lengths vary per step/rank (no static shapes → compile/MX off) | correctness + balance | seed fix applied 19:25; fixed ctx/tgt lengths = design decision (Hugo) |
| PP7 | multi-scale SAM2 | prompt encoder / decoder sized on `datasets.patch_shape`; decoder src = coarsest Hiera level | crash with one patch_shape for both | **config**: Hiera patchifies at `datasets.patch_shape` 4³, SAM-side `meta_arch.sam.patch_shape` / `memory_attention_args.patch_shape` 16³ (the ABC multiscale base already does this); ported into `sam2_ms_base` |

Reflections: (fill at the end)

## 5. Config notes

- `sweep/local_recipe_base.yaml` = `recipe_r1` + local launcher + 200-step probe hooks; every SAM2 leaf inherits it.
- Channel fusions (Hugo 20:40): per-channel patch tokens + channel embedding, then `attn_pool` (one attention query per patch over its C tokens → N tokens; benched = joint cost) or `concat` (C·N tokens through the ViT, mean-pooled at the neck; 5.4× cost). (Hugo 20:50: no plain-average fusion wanted; attn_pool and concat are the two variants.)
- Multi-scale: Hydra cannot `override` groups the bring-up base pulled in by path, so `sam2_ms_base.yaml` merges
  `base_multi_scale`, `masked_hiera_encoder/large_multiscale`, `sam_backbone_hiera_multiscale`, `optimizations/.../large_hiera`
  on top and restates the recipe's model keys (which `base_multi_scale` resets). Same trick for ViT-H (`masked_encoder/huge`).
- Pretraining bases (`pretrain/mae_vit_base`, `jepa_vit_base`): torch-native FSDP2 (`/parallelism/{mae,jepa}`, DCP,
  `adamw_torch`, TORCHTITAN hooks; JEPA adds `ema_scheduler`), DB cube query as SAM2 (`sample_type: cube`, 5 channels,
  synthetic, `has_annotations: false` = annotated and unannotated cubes), layout ZYXC `[128, 384, 512, 5]`, patch 16³
  (6144 tokens, the SAM2 fine-tune grid). Hiera leaves merge `mae|jepa/large_hiera` + `hiera_mu` masking.
