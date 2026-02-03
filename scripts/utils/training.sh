#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/utils/training.sh

# CFG="test_pretrain_4d_mae_local.yaml"
# CFG="pretrain_jepa_local.yaml"
# CFG="experiments/abc/pretrain_mae_test_dali_07_13_2025.yaml"
# CFG="experiments/abc/pretrain_mae_test_torch_07_14_2025.yaml"
# CFG="benchmarks/abc/benchmark_training_4d.yaml"
# CFG="experiments/abc/pretrain_mae_test_tune_07_18_2025.yaml"
# CFG="experiments/abc/pretrain_mae_improve_utilization_07_23_2025.yaml"
# CFG="benchmarks/abc/benchmark_training_4d_dataloader.yaml"
# CFG="benchmarks/abc/benchmark_scaling_4d_base.yaml"

# python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# --- Janelia 

# LSF interactive example cmd:
# bsub -Is -q gpu_h100_parallel -J "debug_job" -n 96 -gpu "num=8:mode=shared" -o "/groups/betzig/betziglab/hph/cell_observatory_project/log.%J" /bin/bash

# micromamba activate
# export PYTHONPATH="/groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform"

# micromamba activate
# export PYTHONPATH="/groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform"

# USAGE: bash /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/scripts/utils/training.sh

# CFG="benchmarks/janelia/benchmark_training_dataloader.yaml"
# CFG="benchmarks/janelia/exp_10_22_2025_mae_3d_batch_size.yaml"
# CFG="experiments/janelia/exp_10_22_2025_hparam_sweep_mae/input_size_128x128x128_X_lr_X_masking_sweep.yaml"

# Janelia
# python3 /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# --- CoreWeave 

# --- MAE

# --- OLD EXPERIMENTS ---

# CFG="experiments/coreweave/tests/exp_11_05_2025_mae_3d_pretrain.yaml"
# CFG="experiments/coreweave/tests/exp_11_05_2025_mae_3d_pretrain_test_sweep.yaml"
# CFG="experiments/coreweave/tests/exp_11_08_25_test_interactive.yaml"

# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p01_X_0p6.yaml"
# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p05_X_0p6.yaml"
# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p001_X_0p6.yaml"

# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p01_X_0p7.yaml"
# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p05_X_0p7.yaml"
# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p001_X_0p7.yaml"

# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p01_X_0p8.yaml"
# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p05_X_0p8.yaml"
# CFG="experiments/coreweave/exp_11_08_25_mae_lr_X_masking_ratio/exp_11_08_25_0p001_X_0p8.yaml"

# CFG="experiments/coreweave/exp_11_11_25_mae_lr_X_masking_ratio_fourier_loss/exp_11_08_25_0p01_X_0p7.yaml"
# CFG="experiments/coreweave/exp_11_11_25_mae_lr_X_masking_ratio_fourier_loss/exp_11_08_25_0p001_X_0p7.yaml"

# CFG="experiments/coreweave/tests/exp_11_13_25_test_inference.yaml"

# CFG="experiments/coreweave/exp_11_15_25_mae_ps_8_fourier_loss/exp_11_15_25_0p0005_X_0p7.yaml"
# CFG="experiments/coreweave/exp_11_15_25_mae_ps_8_fourier_loss/exp_11_15_25_0p002_X_0p7_ps16.yaml"

# ---- ---- ------- 

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p7_X_lamb.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p00001_X_0p7_X_lamb.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p000001_X_0p7_X_lamb.yaml"

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_op7_X_adamw.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p00001_X_op7_X_adamw.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p000001_X_op7_X_adamw.yaml"

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p00001_X_op7_X_lion.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p000001_X_op7_X_lion.yaml"

# --- JEPA

# CFG="experiments/coreweave/exp_11_10_25_jepa_lr_X_masking_ratio/exp_11_08_25_0p01_X_0p6.yaml"
# CFG="experiments/coreweave/exp_11_10_25_jepa_lr_X_masking_ratio/exp_11_08_25_0p01_X_0p7.yaml"
# CFG="experiments/coreweave/exp_11_10_25_jepa_lr_X_masking_ratio/exp_11_08_25_0p01_X_0p8.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p00001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p00001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p00001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p00001_mask_lat_0p7_op8_ax_0p7_op8_opt_lamb.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p0001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p00001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb_h200.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p00001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb_h200.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p0001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p0001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p0001_mask_lat_0p7_op8_ax_0p7_op8_opt_lamb_h200.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op8_ax_0p7_op8_opt_lamb_h200.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p01_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p01_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p01_mask_lat_0p7_op8_ax_0p7_op8_opt_lamb_h200.yaml"

# --- filters

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p7_X_lamb_cdf_target_90_cdf_threshold_150_h200_2M.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p7_X_lamb_cdf_target_90_cdf_threshold_150_h200.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p7_X_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_90_cdf_threshold_150_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_90_cdf_threshold_150_h200_2M.yaml"

# TO RUN
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op8_ax_0p7_op8_opt_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p8_op9_ax_0p8_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p001_X_0p7_X_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p01_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p1_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200.yaml"

# ---- masking ratios (TO RUN)

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p6_X_lamb.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p8_X_lamb.yaml"
# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p9_X_lamb.yaml"

# ---- Finetune (TO RUN)

# channel split

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_baseline_lamb_opt_lr_0p001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_baseline_lamb_opt_lr_0p0001.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_mae_masking_p07_lamb_opt_lr_0p0001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_mae_masking_p07_lamb_opt_lr_0p001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_mae_masking_p07_lamb_opt_lr_0p01.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_jepa_masking_lat_0p7_op9_ax_0p7_op9_lr_0p0001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_jepa_masking_lat_0p7_op9_ax_0p7_op9_lr_0p001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_jepa_masking_lat_0p7_op9_ax_0p7_op9_lr_0p01.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_jepa_masking_lat_0p8_op9_ax_0p8_op9_lr_0p0001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/channel_split_jepa_masking_lat_0p7_op8_ax_0p7_op8_lr_0p0001.yaml"

# upsampling

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_baseline_lr_0p001_h200.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_baseline_lr_0p01_h200.yaml"

# TO RUN MAE
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_space_mae_masking_p07_lamb_opt_lr_0p001_fourier_loss_opt_lamb.yaml"

# TO RUN JEPA
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_space_jepa_masking_lat_0p7_op9_ax_0p7_op9_lr_0p001_fourier_loss_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_space_jepa_masking_lat_0p7_op8_ax_0p7_op8_lr_0p001_fourier_loss_opt_lamb_h200.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_space_jepa_masking_lat_0p8_op9_ax_0p8_op9_lr_0p001_fourier_loss_opt_lamb_h200.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsampling_mae_masking_p07_lamb_opt_lr_0p0001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/upsample_space_mae_masking_p09_lamb_opt_lr_0p001_fourier_loss_opt_lamb.yaml"

# detection

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_no_denoise.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_denoise_200q.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_no_denoise.yaml"

# TO RUN (x5)
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_denoise_200q.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_denoise_200q_2_classes.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_no_denoise_2_classes.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_no_denoise_no_aux_loss.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_denoise_200q_no_aux_loss.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_denoise_200q_no_aux_loss_h100.yaml"

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


# denoise 200q + no aux loss e1-3, 1e-4
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_denoise_200q_no_aux_loss.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_denoise_200q_no_aux_loss.yaml"

# no denoise + no aux loss 1e-3
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_no_denoise_no_aux_loss.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p0001_denoise_400q_no_aux_loss.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_no_denoise_w_aux_loss.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_denoise_200q_w_aux_loss.yaml"

# no denoise, aux loss 1e-2

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p01_no_denoise_w_aux_loss.yaml"

# TO RUN:

# denoise (1000q) + aux loss 1e-3
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/instance_segmentation_plainDETR_baseline_denoise_lr_0p001_denoise_1000q_w_aux_loss.yaml"

# MAE, no denoise, aux loss 1e-3
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_mae_masking_0p9_denoise_lr_0p001_no_denoise_w_aux_loss.yaml"

# JEPA, no denoise, aux loss 1e-3
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p001_no_denoise_w_aux_loss.yaml"

# JEPA, no denoise, aux loss 5e-4
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p0005_no_denoise_w_aux_loss.yaml"

# JEPA, no denoise, aux loss 1e-4
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p0001_no_denoise_w_aux_loss.yaml"

# JEPA no denoise, aux loss 1e-5
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p00001_no_denoise_w_aux_loss.yaml"

# =======

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200_w_hook.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p8_0p9_ax_0p8_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200_w_hook.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p9_op9_ax_0p9_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200_w_hook.yaml"
# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_1p0_1p0_ax_0p7_op7_opt_lamb_cdf_target_80_cdf_threshold_100_h200_w_hook.yaml"

# CFG="experiments\coreweave\exp_12_28_25_jepa_lr_X_masking_ratio\lr_0p001_mask_lat_0p9_op9_ax_0p9_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200_w_hook_no_rope.yaml"

# CFG="experiments/coreweave/exp_12_28_25_jepa_lr_X_masking_ratio/lr_0p001_mask_lat_0p7_op9_ax_0p7_op9_opt_lamb_cdf_target_80_cdf_threshold_100_h200_w_hook_no_rope.yaml"

# CFG="experiments/coreweave/exp_12_15_25_mae_lr_X_masking_ratio_fourier_loss/0p0001_X_0p7_X_lamb_cdf_target_90_cdf_threshold_150_h200_no_rope.yaml"
# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p001_no_denoise_w_aux_loss_no_rope.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p0001_no_denoise_w_aux_loss_no_rope.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/maskDINO_baseline_lr_0p001.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_jepa_masking_lat_0p7_op9_ax_0p7_op9_denoise_lr_0p01_no_denoise_w_aux_loss_no_rope.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_mae_masking_0p7_denoise_lr_0p001_no_denoise_w_aux_loss.yaml"

CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/plainDETR_mae_masking_0p7_denoise_lr_0p001_denoise_w_aux_loss.yaml"

# CFG="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task/maskDINO_mae_masking_0p7_no_rope_lr_0p001.yaml"

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# inference

# CFG="experiments/coreweave/tests/test_feature_viz_mae.yaml"
# CFG="experiments/coreweave/tests/test_feature_viz_jepa.yaml"

# --- Linux

# USAGE: bash /work/cell_observatory_platform/scripts/utils/training.sh

# python3 /work/cell_observatory_platform/manager.py --config-name=${CFG}

# --- Windows

# USAGE: & "$Env:ProgramFiles\Git\bin\bash.exe" -lc '"/c/Users/HugoPatricHamilton/git_managed/cell-observatory/cell_observatory_platform/scripts/utils/training.sh"'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

MANAGER_PY="$REPO_ROOT/cell_observatory_platform/manager.py"
if command -v cygpath >/dev/null 2>&1; then
  MANAGER_PY="$(cygpath -u "$MANAGER_PY")"
fi

echo "[training.sh] Repo root: $REPO_ROOT"
echo "[training.sh] Manager:   $MANAGER_PY"
echo "[training.sh] Config:    $CFG"

if command -v uv >/dev/null 2>&1; then
  exec uv run python "$MANAGER_PY" --config-name="$CFG"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 "$MANAGER_PY" --config-name="$CFG"
else
  exec python "$MANAGER_PY" --config-name="$CFG"
fi

# ALL JOBS: ai job list --proj cell-observatory
# LOGS: ai job follow --name <job_name> --project cell-observatory