"""
Adapted from:
https://github.com/facebookresearch/dinov3/dinov3/train/ssl_meta_arch.py
"""

import gc
import inspect
import logging
from typing import Tuple, Mapping, Any, Literal
from functools import partial

import torch
from torch import Tensor, nn

from omegaconf import OmegaConf

from cell_observatory_platform.models.heads.linear_head import LinearHead
from cell_observatory_platform.models.backbones.dino_encoder import DinoEncoder
from cell_observatory_platform.training.schedulers import linear_warmup_cosine_decay
from cell_observatory_platform.training.losses import (
    DINOLoss, 
    KoLeoLoss, 
    KoLeoLossDistributed, 
    iBOTPatchLoss
)

# from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
# from dinov3.utils import count_parameters


class DINO(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        # backbones
        student_backbone: nn.Module,
        teacher_backbone: nn.Module,
        # heads
        ibot_separate_head: bool,
        centering: str,
        dino_head_n_prototypes: int,
        dino_head_hidden_dim: int,
        dino_head_bottleneck_dim: int,
        dino_head_nlayers: int,
        ibot_head_n_prototypes: int,
        ibot_head_hidden_dim: int,
        ibot_head_bottleneck_dim: int,
        ibot_head_nlayers: int,
        ibot_mask_ratio_min_max: Tuple[float, float],
        ibot_mask_sample_probability: float,
        # losses
        koleo_loss_distributed: KoLeoLossDistributed,
        koleo_distributed_replicas: int,
        koleo_topk: int,
        dino_global_ignore_diagonal: bool,
        dino_loss_weight: float,
        dino_koleo_loss_weight: float,
        ibot_loss_weight: float,
        koleo_distributed_loss_group_size: int,
        reweight_dino_local_loss: bool,
        local_crops_number: int,
        # global_crops_size: int,
    ):
        super().__init__()

        assert ibot_separate_head is True
        assert centering == "sinkhorn_knopp"

        student_model_dict = dict()
        teacher_model_dict = dict()

        # NOTE: see BUILD function for more details
        student_backbone = student_backbone
        teacher_backbone = teacher_backbone
        embed_dim = embed_dim
        
        torch.cuda.empty_cache()
        gc.collect()

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        self.embed_dim = embed_dim  # D
        self.dino_output_dim = dino_head_n_prototypes  # K

        head = partial(
            LinearHead,
            in_dim=embed_dim,
            output_dim=dino_head_n_prototypes,
            hidden_dim=dino_head_hidden_dim,
            bottleneck_dim=dino_head_bottleneck_dim,
            nlayers=dino_head_nlayers,
        )
        student_model_dict["dino_head"] = head()
        teacher_model_dict["dino_head"] = head()
        
        self.dino_loss = DINOLoss(self.dino_output_dim)

        if koleo_loss_distributed:
            assert koleo_distributed_replicas == 0, (
                "Option `koleo_distributed_replicas` is no longer supported"
            )
            self.koleo_loss = KoLeoLossDistributed(
                topk=koleo_topk,
                loss_group_size=koleo_distributed_loss_group_size,
            )
        else:
            assert koleo_topk == 1, "Non-distributed KoLeo loss only supports `koleo_topk=1`"
            self.koleo_loss = KoLeoLoss()

        assert 0 <= ibot_mask_ratio_min_max[0] < ibot_mask_ratio_min_max[1] <= 1, (
            "provide a valid ibot_mask_ratio_min_max"
        )
        assert 0 <= ibot_mask_sample_probability <= 1, "provide a positive mask probability for ibot"
        ibot_head_class = partial(
            LinearHead,
            in_dim=embed_dim,
            output_dim=ibot_head_n_prototypes,
            hidden_dim=ibot_head_hidden_dim,
            bottleneck_dim=ibot_head_bottleneck_dim,
            nlayers=ibot_head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(ibot_head_n_prototypes)

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # NOTE: this may be overwritten for distillation
        self.model_ema = self.teacher

        # NOTE: no grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        # NOTE: set config params
        self.n_local_crops = local_crops_number
        self.dino_global_ignore_diagonal = dino_global_ignore_diagonal
        self.dino_loss_weight = dino_loss_weight
        self.dino_koleo_loss_weight = dino_koleo_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.reweight_dino_local_loss = reweight_dino_local_loss

        # NOTE: Local loss reweighting
        if self.reweight_dino_local_loss:
            self.dino_local_loss_schedule = None
            # self.student_crop_size = global_crops_size

        # NOTE: will be set by TeacherTemperatureSchedulerHook (see training/hooks.py)
        self.teacher_temperature = None

    def init_model_weights(self, buffer_device: str | None = None) -> None:
        # NOTE: all weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        # FIXME: we currently don't support this, but we should add it in the future
        # if self.resume_from_teacher_chkpt:
        #     logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
        #     init_fsdp_model_from_checkpoint(
        #         self.student,
        #         self.cfg.student.resume_from_teacher_chkpt,
        #         skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
        #         keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
        #         process_group=distributed.get_process_subgroup(),
        #     )
        #     self.model_ema.load_state_dict(self.student.state_dict())
        self.teacher.backbone.init_weights()
        self.teacher.dino_head.init_weights()
        self.teacher.ibot_head.init_weights()

    def forward(self, data_sample: dict) -> tuple[Tensor, dict[str, float | Tensor]]:
        data_tensors = data_sample["data_tensors"]
        local_meta = data_sample["metainfo"]["dataset_stream_metainfo"]["local_crops"]
        global_meta = data_sample["metainfo"]["dataset_stream_metainfo"]["global_crops"]

        global_crops = data_tensors["global_crops"].cuda(non_blocking=True)
        local_crops = data_tensors["local_crops"].cuda(non_blocking=True)
        masks = global_meta["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = global_meta["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = global_meta["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = global_meta["n_masked_patches"].cuda(non_blocking=True)
        upperbound = global_meta["upperbound"]

        n_global_crops = global_meta["n_crops"]
        n_local_crops = local_meta["n_crops"]
        local_batch_size = data_sample["metainfo"]["local_batch_size"]
        global_batch_size = data_sample["metainfo"]["global_batch_size"]
        assert global_crops.shape[0] == n_global_crops * local_batch_size, "global_crops.shape[0] != n_global_crops * local_batch_size"

        metrics_dict = {}
        metrics_dict["local_batch_size"] = local_batch_size
        metrics_dict["global_batch_size"] = global_batch_size

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.forward_features_teacher(
            data_tensors=global_crops.unflatten(0, (n_global_crops, local_batch_size)),
            teacher_temp=self.teacher_temperature,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=upperbound,
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.forward_features_student(
            global_crops=global_crops.unflatten(0, (n_global_crops, local_batch_size)),
            local_crops=local_crops.unflatten(0, (n_local_crops, local_batch_size)),
            upperbound=upperbound,
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=data_sample["metainfo"]["idx"],
        )

        loss_dict["step_loss"] = loss_accumulator

        return loss_dict, None

    @torch.no_grad()
    def forward_features_teacher(
        self,
        data_tensors,
        upperbound,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
    ):
        n_crops, B, *_ = data_tensors.shape
        data_tensors = data_tensors.flatten(0, 1)

        backbone_out = self.teacher.backbone(data_tensors, is_training=True)
        # [n_crops * B, D]
        class_tokens = backbone_out["x_norm_clstoken"]
        # [n_crops * B, R, D]
        register_tokens = backbone_out["x_storage_tokens"]
        # [n_crops * B, P, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        # [n_crops * B, K]
        class_tokens_after_head = self.teacher.dino_head(class_tokens)

        # Center with sinkhorn-knopp
        # [n_crops * B, K]
        class_tokens_centered = self.dino_loss.sinkhorn_knopp_teacher(
            class_tokens_after_head, teacher_temp=teacher_temp
        )
        # [n_crops, B, K]
        class_tokens_centered = class_tokens_centered.unflatten(0, (n_crops, B))
        # [n_masked_patches, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )

        return {
            # [n_crops, B, D]
            "cls_pre_head": class_tokens.unflatten(0, [n_crops, B]),
            # [n_crops, B, R, D]
            "reg_pre_head": register_tokens.unflatten(0, [n_crops, B]),
            # [n_crops, B, P, D]
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),
            # [n_crops, B, K]
            "cls_after_head": class_tokens_after_head.unflatten(0, [n_crops, B]),
            # [n_crops, B, K]
            "cls_centered": class_tokens_centered,
            # [n_masked_patches, K]
            "masked_patch_centered": masked_patch_centered,
        }

    def forward_features_student(
        self, 
        global_crops, 
        local_crops, 
        upperbound, 
        masks, 
        mask_indices_list
    ):
        n_global_crops, B, *_ = global_crops.shape
        n_local_crops, B, *_ = local_crops.shape

        global_crops = global_crops.flatten(0, 1)
        local_crops = local_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops],
            masks=[masks, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_losses(
        self,
        teacher_global,
        student_global,
        student_local,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        
        loss_dict = {}
        loss_accumulator = 0.0

        # NOTE: Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        return loss_accumulator, loss_dict

    def train(self):
        super().train()
        self.teacher.eval()

    # see training/hooks.py EMASchedulerHook
    def ema_update(self, beta):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, beta)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - beta)

    # TODO: move to training/helpers.py
    # def get_maybe_fused_params_for_submodel(self, m: nn.Module):
    #     params_groups = get_params_groups_with_decay_fsdp(
    #         model=m,
    #         lr_decay_rate=self.cfg.optim.layerwise_decay,
    #         patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
    #         dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
    #     )
    #     if self.cfg.optim.multi_tensor_optim:
    #         fused_params_groups = fuse_params_groups(params_groups)
    #         logger.info("fusing param groups")

    #         for g in fused_params_groups:
    #             g["foreach"] = True
    #             g["fused"] = True
    #         return fused_params_groups
    #     else:
    #         return params_groups

    # def get_params_groups(self):
    #     all_params_groups = []
    #     for name, m in self.student.items():
    #         logger.info(f"Getting paramer groups for {name}")
    #         all_params_groups += self.get_maybe_fused_params_for_submodel(m)
    #     return all_params_groups

    # TODO: implement in ParallelEpochBasedTrainer
    # def prepare_for_distributed_training(self) -> None:
    #     process_subgroup = distributed.get_process_subgroup()
    #     default_process_group = distributed.get_default_process_group()
    #     inference_only_models = [self.model_ema]
    #     inference_only_models_process_groups = [process_subgroup]
        # ac_compile_parallelize(
        #     trained_model=self.student,
        #     inference_only_models=inference_only_models,
        #     cfg=self.cfg,
        #     trained_model_process_group=process_subgroup,
        #     inference_only_models_process_groups=inference_only_models_process_groups,
        # )


def _extract_backbone_kwargs(cfg: Mapping[str, Any]) -> dict:
    """Extract kwargs for DinoEncoder from the backbone config."""
    
    sig = inspect.signature(DinoEncoder.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "ignored_kwargs", "device"}
    ignore = {"_target_", "BUILD"}
    kwargs = {}
    for k in cfg.keys():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = cfg[k]
    return kwargs


def BUILD(cfg: Mapping[str, Any]) -> DINO:
    meta_cfg = cfg.models.meta_arch.dino
    backbone_cfg = cfg.models.backbones.dino_encoder

    # Resolve configs so interpolations are resolved
    backbone_cfg_resolved = OmegaConf.to_container(backbone_cfg, resolve=True)
    meta_cfg_resolved = OmegaConf.to_container(meta_cfg, resolve=True)
    meta = meta_cfg_resolved
    backbone_kwargs = _extract_backbone_kwargs(backbone_cfg_resolved)

    # Build student and teacher backbones (identical architecture, separate instances)
    student_backbone = DinoEncoder(**backbone_kwargs)
    teacher_backbone = DinoEncoder(**backbone_kwargs)
    embed_dim = student_backbone.embed_dim

    # koleo_loss_distributed is used as a bool in __init__
    koleo_loss_distributed = meta.get("koleo_loss_distributed", False)
    koleo_distributed_loss_group_size = meta.get("koleo_distributed_loss_group_size")

    ibot_mask_ratio_min_max = meta["ibot_mask_ratio_min_max"]
    if not isinstance(ibot_mask_ratio_min_max, tuple):
        ibot_mask_ratio_min_max = tuple(ibot_mask_ratio_min_max)

    return DINO(
        embed_dim=embed_dim,
        student_backbone=student_backbone,
        teacher_backbone=teacher_backbone,
        ibot_separate_head=meta["ibot_separate_head"],
        centering=meta["centering"],
        dino_head_n_prototypes=meta["dino_head_n_prototypes"],
        dino_head_hidden_dim=meta["dino_head_hidden_dim"],
        dino_head_bottleneck_dim=meta["dino_head_bottleneck_dim"],
        dino_head_nlayers=meta["dino_head_nlayers"],
        ibot_head_n_prototypes=meta["ibot_head_n_prototypes"],
        ibot_head_hidden_dim=meta["ibot_head_hidden_dim"],
        ibot_head_bottleneck_dim=meta["ibot_head_bottleneck_dim"],
        ibot_head_nlayers=meta["ibot_head_nlayers"],
        ibot_mask_ratio_min_max=ibot_mask_ratio_min_max,
        ibot_mask_sample_probability=meta["ibot_mask_sample_probability"],
        koleo_loss_distributed=koleo_loss_distributed,
        koleo_distributed_replicas=meta.get("koleo_distributed_replicas", 0),
        koleo_topk=meta.get("koleo_topk", 1),
        dino_global_ignore_diagonal=meta["dino_global_ignore_diagonal"],
        dino_loss_weight=meta["dino_loss_weight"],
        dino_koleo_loss_weight=meta["dino_koleo_loss_weight"],
        ibot_loss_weight=meta["ibot_loss_weight"],
        koleo_distributed_loss_group_size=koleo_distributed_loss_group_size,
        reweight_dino_local_loss=meta.get("reweight_dino_local_loss", False),
        local_crops_number=meta["local_crops_number"],
    )