import logging
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

import ray
import torch
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.training.checkpoint_metadata import (
    metadata_path_for_tag,
    read_metadata_json,
    write_metadata_json,
)
from cell_observatory_platform.utils.context import barrier, is_main_process

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        model: torch.nn.Module,
        zero_stage: int,
        load_universal_checkpoint: bool,
        save_checkpointdir: Union[str, Path],
        save_period: int = 1,
        load_optimizer: bool = True,
        load_dtype: Optional[str] = None,
        resume_checkpointdir: Optional[Union[str, Path]] = None,
        pretrained_checkpointdir: Optional[Union[str, Path]] = None,
        backend: Literal["DEEPSPEED"] = "DEEPSPEED",
        checkpoint_tag: str = "best_model",
        use_custom_state_dict_filter: Optional[List[str]] = None,
        ckpt_include_prefixes: Optional[List[str]] = None,
        ckpt_translate_map: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.backend = backend.upper()
        self.save_period = save_period
        self.zero_stage = zero_stage
        self.load_dtype = load_dtype
        self.checkpoint_tag = checkpoint_tag
        self.load_universal_checkpoint = load_universal_checkpoint

        self.ckpt_translate_map = ckpt_translate_map
        self.ckpt_include_prefixes = ckpt_include_prefixes

        assert not (resume_checkpointdir is not None and pretrained_checkpointdir is not None), (
            "Cannot specify both `resume_checkpointdir` and `pretrained_checkpointdir`. "
            "Please choose one of them or neither."
        )

        self.resume_checkpointdir = resume_checkpointdir
        self.pretrained_checkpointdir = pretrained_checkpointdir

        # NOTE: if the user does not explicitly set load_optimizer,
        #       we default to loading optimizer state only when resuming
        if load_optimizer is not None:
            logger.info("[CheckpointManager] loading optimizer state with DeepSpeed.")
            self.load_optimizer = load_optimizer
        else:
            self.load_optimizer = self.resume_checkpointdir is not None
            logger.info(f"[CheckpointManager] `load_optimizer` set to {self.load_optimizer}.")

        self.load_scheduler, self.load_module_only = self.load_optimizer, not self.load_optimizer

        if resume_checkpointdir is not None:
            self.load_checkpointdir = Path(resume_checkpointdir)
        elif pretrained_checkpointdir is not None:
            self.load_checkpointdir = Path(pretrained_checkpointdir)
        else:
            self.load_checkpointdir = None

        self.save_checkpointdir = (
            Path(save_checkpointdir) if isinstance(save_checkpointdir, str) else save_checkpointdir
        )

        self.use_custom_state_dict_filter = use_custom_state_dict_filter

    def save(
        self,
        prefix: str,
        save_epoch: int = None,
        save_step: int = None,
        save_best_loss: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        self.save_checkpointdir.mkdir(parents=True, exist_ok=True)
        if self.backend == "DEEPSPEED":
            if metadata is None:
                raise ValueError(
                    "CheckpointManager.save() requires `metadata` dict "
                    "(use training.checkpoint_metadata.build_metadata)."
                )
            client_state = dict(metadata)
            self.model.save_checkpoint(self.save_checkpointdir, client_state=client_state, tag=prefix)
            meta_path = metadata_path_for_tag(self.save_checkpointdir, prefix)
            if is_main_process():
                write_metadata_json(meta_path, metadata)
            barrier()
        else:
            # When implementing TORCH/plain save, write checkpoint_meta.json atomically next to weights.
            raise NotImplementedError("Saving checkpoints for " "other backends not implemented yet.")

    def load(self):
        if self.resume_checkpointdir is not None:
            return self._load_deepspeed()
        elif self.pretrained_checkpointdir is not None:
            return self._load_torch()
        else:
            raise ValueError(
                "No checkpoint directory specified for loading. "
                "Please set `resume_checkpointdir` or `pretrained_checkpointdir`."
            )

    def _load_deepspeed(self):
        assert self.load_checkpointdir is not None, (
            "No checkpoint directory specified for loading. "
            "Please set `resume_checkpointdir` or `pretrained_checkpointdir`."
        )

        ckpt_zero_stage = self._get_zero_stage(os.path.join(self.load_checkpointdir, self.checkpoint_tag))

        if self.use_custom_state_dict_filter is not None and self.backend == "DEEPSPEED":
            custom_load_fn = self.make_state_dict_filter_fn(
                include_prefixes=self.ckpt_include_prefixes, translate_map=self.ckpt_translate_map
            )
        else:
            custom_load_fn = None

        # we do not currently support loading ZeRO-0 checkpoints
        # into ZeRO-1/2/3 models
        if (self.zero_stage == 0 and ckpt_zero_stage != 0) or (self.zero_stage != 0 and ckpt_zero_stage == 0):
            raise ValueError(
                f"Cannot load a ZeRO-0 checkpoint into a \
                    ZeRO-{self.zero_stage} model or vice versa. "
            )

        # if we are resuming from a checkpoint that has an identical zero stage
        # and zero configuration, which is specified by setting
        # `load_universal_checkpoint` to False and `resume_checkpointdir` to a valid path
        # we can load the checkpoint directly with the load_checkpoint API
        if self.resume_checkpointdir is not None and not self.load_universal_checkpoint:
            ckpt_path, _ = self._load_checkpoint(tag=self.checkpoint_tag, custom_load_fn=custom_load_fn)

        # if we are resuming from a checkpoint that has a different zero configuration
        # we first attempt to find an existing universal checkpoint to load.
        # if it does not exist, we convert the ZeRO-1/2/3 checkpoint to a universal
        # checkpoint and then load it. this is the new prefered way to ensure that
        # the state of the model is loaded correctly when restarting a job with a different
        # zero stage or configuration. see: https://arxiv.org/pdf/2406.18820
        else:
            assert self.model.load_universal_checkpoint(), (
                "Model does not support loading universal checkpoints. "
                "Please set `allow_universal_conversion` to True in "
                "DeepSpeed Config."
            )

            universal_checkpointdir = self.load_checkpointdir / f"{self.checkpoint_tag}_universal"

            # NOTE: disabling reuse of existing universal checkpoints since this may lead to
            #       to that the wrong (old) checkpoint being loaded if we need >1 restarts.
            # if universal_checkpointdir.exists() and any(universal_checkpointdir.iterdir()):
            #     logger.info(
            #         f"Loading universal checkpoint from existing directory \
            #             {self.load_checkpointdir / f'{self.checkpoint_tag}_universal'}"
            #     )
            #     ckpt_path, client_state = self._load_checkpoint(
            #         tag=f"{self.checkpoint_tag}_universal", custom_load_fn=custom_load_fn
            #     )

            # else:

            logger.info(
                f"Converting ZeRO-0 checkpoint to universal format: \
                {universal_checkpointdir}"
                "NOTE: this will create a new checkpoint \
                with the tag `{self.checkpoint_tag}_universal` in the same directory."
            )

            if is_main_process():
                self._convert_zero_checkpoint_to_universal(
                    src=self.load_checkpointdir / self.checkpoint_tag,
                    dst=self.load_checkpointdir / f"{self.checkpoint_tag}_universal",
                )

            barrier()

            ckpt_path, _ = self._load_checkpoint(
                tag=f"{self.checkpoint_tag}_universal", custom_load_fn=custom_load_fn
            )

        # get target dtype if specified
        if self.load_dtype is not None:
            module = getattr(self.model, "module", self.model)
            module.to(TORCH_DTYPES[self.load_dtype].value)

        meta_path = metadata_path_for_tag(self.load_checkpointdir, self.checkpoint_tag)
        checkpoint_meta = read_metadata_json(meta_path, allow_missing=True)
        return ckpt_path, checkpoint_meta

    def _load_torch(self, checkpoint: str = "mp_rank_00_model_states.pt"):
        tag_dir = self.load_checkpointdir / self.checkpoint_tag
        pt_path = tag_dir / checkpoint
        sd = torch.load(pt_path, map_location="cpu")

        src = None
        for k in ("module", "state_dict", "model", "model_state_dict"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                src = sd[k]
                break
        if src is None:
            raise ValueError("Could not find state_dict in checkpoint.")

        # ensure prefix matches destination
        dst_module = getattr(self.model, "module", self.model)
        dst_module = getattr(dst_module, "_orig_mod", dst_module)
        if self.use_custom_state_dict_filter:
            custom_load_fn = self.make_state_dict_filter_fn(
                include_prefixes=self.ckpt_include_prefixes, translate_map=self.ckpt_translate_map
            )
            custom_load_fn(src, dst_module)
        else:
            src = self._prefix_aware_load_state_dict(src, dst_module)
            src = self._strip_torch_compile_state_dict(src)
            missing, unexpected = dst_module.load_state_dict(src, strict=False)
            if missing:
                logger.info("[CheckpointManager] (torch) missing keys: %s", list(missing))
            if unexpected:
                logger.info("[CheckpointManager] (torch) unexpected keys: %s", list(unexpected))

        # optional dtype cast
        if self.load_dtype is not None:
            dst_module.to(TORCH_DTYPES[self.load_dtype].value)

        meta_path = metadata_path_for_tag(self.load_checkpointdir, self.checkpoint_tag)
        checkpoint_meta = read_metadata_json(meta_path, allow_missing=True)
        return str(pt_path), checkpoint_meta

    def _load_checkpoint(self, tag: str, custom_load_fn=None):
        return self.model.load_checkpoint(
            self.load_checkpointdir,
            tag=tag,
            load_optimizer_states=self.load_optimizer,
            load_lr_scheduler_states=self.load_scheduler,
            load_module_only=self.load_module_only,
            custom_load_fn=custom_load_fn,
        )

    def make_state_dict_filter_fn(
        self,
        include_prefixes: Optional[Iterable[str]] = None,
        translate_map: Optional[Dict[str, str]] = None,
    ):
        """
        Returns custom_load_fn(src_state_dict, dst_module)
        that:
        - translates keys via translate_map,
        - filters to keys starting with any of include_prefixes (if provided),
        - drops any tensor whose shape doesn't match the destination tensor,
        - logs dropped/missing/unexpected keys,
        - and loads with strict=False so non-matching layers remain at init.
        """
        include_prefixes = list(include_prefixes) if include_prefixes else None
        translate_map = dict(translate_map) if translate_map else {}

        def _translate_key(k: str) -> str:
            # prefix-based rename: old.* -> new.*
            for old, new in translate_map.items():
                if k == old:
                    return new
                if k.startswith(old + "."):
                    return new + k[len(old):]  # keeps the dot + suffix
            return k

        def custom_load_fn(src: Dict[str, torch.Tensor], dst: torch.nn.Module):
            # if dst is compiled, load into the original module
            dst = getattr(dst, "_orig_mod", dst)
            dst_state_dict = dst.state_dict()
            src_state_dict = self._prefix_aware_load_state_dict(src, dst)
            src_state_dict = self._strip_torch_compile_state_dict(src_state_dict)

            # apply key translations
            if translate_map:
                translated = {}
                for k, v in src_state_dict.items():
                    nk = _translate_key(k)
                    if nk in translated:
                        raise ValueError(f"Key translation map results in duplicate key: {nk}")
                    translated[nk] = v
                src_state_dict = translated

            if include_prefixes:

                def _in_prefix_list(k: str) -> bool:
                    return any(k.startswith(pref) for pref in include_prefixes)

                src_state_dict = {k: v for k, v in src_state_dict.items() if _in_prefix_list(k)}

            keep, dropped = {}, []
            for k, v in src_state_dict.items():
                dst_t = dst_state_dict.get(k, None)
                if dst_t is not None and tuple(v.shape) == tuple(dst_t.shape):
                    keep[k] = v
                else:
                    dropped.append((k, tuple(v.shape), tuple(dst_t.shape) if dst_t is not None else None))

            # NOTE: helpful debug logging
            # ray.logger.warning(
            #     "[CheckpointManager] Src state dict keys: {} | Dst state dict keys: {}",
            #     src_state_dict.keys(),
            #     dst_state_dict.keys(),
            # )

            missing, unexpected = dst.load_state_dict(keep, strict=False)

            if dropped:
                ray.logger.warning(
                    "[CheckpointManager] Dropped %d mismatched tensors (shape or missing):\n%s",
                    len(dropped),
                    "\n".join([f"  - {k}: ckpt{cs} -> model{ms}" for (k, cs, ms) in dropped]),
                )
            if missing:
                ray.logger.info(
                    "[CheckpointManager] Model missing keys after load (left at init): %s", list(missing)
                )
            if unexpected:
                ray.logger.info("[CheckpointManager] Unexpected keys ignored: %s", list(unexpected))

            ray.logger.warning(
                    "[CheckpointManager] Kept %d tensors:\n%s",
                    len(keep),
                    "\n".join([f"  - {k}: {tuple(v.shape)}" for (k, v) in keep.items()]),
            )

        return custom_load_fn
    
    @staticmethod
    def _strip_torch_compile_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        torch.compile may wrap modules and produce keys containing '_orig_mod'.
        Normalize keys so compiled and uncompiled checkpoints can load interchangeably.
        """
        out = OrderedDict()
        for k, v in sd.items():
            nk = k

            # common forms:
            #   "_orig_mod.xxx"
            #   "module._orig_mod.xxx"
            #   "xxx._orig_mod.yyy"
            if nk.startswith("module._orig_mod."):
                nk = "module." + nk[len("module._orig_mod."):]
            if nk.startswith("_orig_mod."):
                nk = nk[len("_orig_mod."):]

            # remove any nested occurrences
            while "._orig_mod." in nk:
                nk = nk.replace("._orig_mod.", ".")

            out[nk] = v
        return out

    def _get_zero_stage(self, ckpt_dir: Union[str, Path]) -> int:
        """Return ZeRO stage (0-3) from any DeepSpeed checkpoint tag folder."""
        tag_dir = Path(ckpt_dir)
        # see https://github.com/deepspeedai/DeepSpeed/deepspeed/checkpoint/constants.py
        # for naming used in DeepSpeed checkpoint nomenclature
        try:
            f = next(tag_dir.glob("*_model_states.pt"))
        except StopIteration:
            raise ValueError(f"No DeepSpeed checkpoint found in {ckpt_dir}")
        z = torch.load(f, map_location="cpu")
        stage = z["ds_config"]["zero_optimization"]["stage"]
        del z  # free memory
        return stage

    def _prefix_aware_load_state_dict(self, state_dict, model):
        ckpt_has_module = any(k.startswith("module.") for k in state_dict)
        model_expects_mod = any(k.startswith("module.") for k in model.state_dict())
        if ckpt_has_module and not model_expects_mod:
            return self._strip_prefix(state_dict, "module.")
        elif not ckpt_has_module and model_expects_mod:
            return self._add_prefix(state_dict, "module.")
        return state_dict

    @staticmethod
    def _add_prefix(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
        """
        If none of the keys start with `prefix`, add it to every key.
        Otherwise return the dict unchanged.
        """
        return OrderedDict((k if k.startswith(prefix) else f"{prefix}{k}", v) for k, v in state_dict.items())

    @classmethod
    def _strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
        """
        If the keys in `state_dict` are all prefixed with `prefix`, remove it.
        Otherwise return the dict unchanged.
        """
        # do all keys start with the prefix?
        if all(key.startswith(prefix) for key in state_dict.keys()):
            new_state = OrderedDict()
            for key, value in state_dict.items():
                new_key = key[len(prefix) :]
                new_state[new_key] = value
            return new_state
        else:
            return state_dict

    def _convert_zero_checkpoint_to_universal(self, src: Path, dst: Path):
        """
        Convert a DeepSpeed Zero Stage 3 checkpoint to a standard checkpoint
        that may be loaded more flexibly.
        """
        if not dst.exists():
            os.makedirs(dst)

        cmd = [
            sys.executable,
            "-m",
            "deepspeed.checkpoint.ds_to_universal",
            "--input_folder",
            str(src),
            "--output_folder",
            str(dst),
            "--inject_missing_state",
        ]

        subprocess.check_call(cmd)

    # useful utility function to convert a deepspeed
    # zero stage 3 checkpoint to a standard checkpoint
    def convert_zero_checkpoint(checkpoint_dir: str, tag: str = "best_model"):
        checkpoint_path = os.path.join(checkpoint_dir, tag)
        output_path = os.path.join(checkpoint_path, f"{tag}_universal")
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir=checkpoint_path, output_dir=output_path)
