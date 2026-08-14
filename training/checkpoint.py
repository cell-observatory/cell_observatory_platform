import logging
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import ray
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.training.checkpoint_metadata import (
    build_metadata,
    default_metadata,
    metadata_path_for_tag,
    read_metadata_json,
    write_metadata_json,
)
from cell_observatory_platform.utils.context import barrier, is_main_process

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        model: torch.nn.Module,
        zero_stage: int,
        load_universal_checkpoint: bool,
        save_checkpointdir: Union[str, Path],
        save_period: int = 1,
        # None -> "load optimizer state only when resuming" (the documented
        # default below); an explicit bool always wins. A `True` default made
        # the else-branch unreachable.
        load_optimizer: Optional[bool] = None,
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

        if resume_checkpointdir is not None and pretrained_checkpointdir is not None:
            # survives python -O (a stripped assert silently accepted both)
            raise ValueError(
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

    def load_for_eval(self, job_type: str):
        """Weight loading for eval-time trainers (TestTrainer / Inferencer).

        Only ``pretrained_checkpointdir`` is valid here: eval loads into the raw
        ``nn.Module`` (no DeepSpeed engine), so a ``resume_checkpointdir`` -- the
        natural reuse of a training config -- would either route to the engine
        loader (AttributeError) or, worse, silently evaluate the random init.
        Fail loud in both wrong configurations.
        """
        if self.pretrained_checkpointdir:
            _, checkpoint_meta = self.load()
            return checkpoint_meta
        if self.resume_checkpointdir:
            raise ValueError(
                f"job_type={job_type} loads weights via pretrained_checkpointdir; "
                "resume_checkpointdir is a DeepSpeed-engine resume dir and cannot "
                "be loaded into the raw nn.Module used at eval/inference time. "
                "Point pretrained_checkpointdir at the checkpoint tag instead."
            )
        raise ValueError(
            f"No checkpoint directory configured for job_type={job_type} -- "
            "refusing to run with randomly initialized weights. Set "
            "checkpoint.checkpoint_manager.pretrained_checkpointdir."
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

        # ZeRO-3 partitions parameters across ranks: the module state dict in
        # mp_rank_00_model_states.pt holds (mostly empty) placeholders, so a
        # plain torch load would silently produce a partially-initialized model.
        if isinstance(sd, dict):
            ckpt_stage = (
                sd.get("ds_config", {}).get("zero_optimization", {}).get("stage", 0)
                if isinstance(sd.get("ds_config", {}), dict) else 0
            )
            if ckpt_stage == 3:
                raise ValueError(
                    f"Refusing to torch-load a ZeRO-3 checkpoint from {pt_path}: "
                    f"parameters are partitioned and the model-states file does not "
                    f"contain the full weights. Convert it first with "
                    f"deepspeed.utils.zero_to_fp32 (see convert_zero_checkpoint)."
                )

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
        - logs dropped/missing/unexpected keys.

        Strictness: when NO include_prefixes and NO translate_map are
        configured, this is a plain resume — any architecture drift (dropped
        shape mismatches, missing or unexpected keys) must FAIL loudly instead
        of silently leaving layers at init. Only a user-configured translation
        opts into the permissive strict=False load.
        """
        include_prefixes = list(include_prefixes) if include_prefixes else None
        translate_map = dict(translate_map) if translate_map else {}
        strict = not (include_prefixes or translate_map)

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

            if strict and dropped:
                raise RuntimeError(
                    "[CheckpointManager] strict resume: checkpoint/model architecture "
                    "drift — %d tensors mismatched in shape or missing from the model:\n%s\n"
                    "Configure ckpt_include_prefixes/ckpt_translate_map to opt into a "
                    "permissive load."
                    % (
                        len(dropped),
                        "\n".join([f"  - {k}: ckpt{cs} -> model{ms}" for (k, cs, ms) in dropped]),
                    )
                )

            # strict=True raises on missing/unexpected keys (surfacing them in the
            # error); the permissive path logs them below instead.
            missing, unexpected = dst.load_state_dict(keep, strict=strict)

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

    @staticmethod
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
    @staticmethod
    def convert_zero_checkpoint(checkpoint_dir: str, tag: str = "best_model"):
        # lazy: keep deepspeed off the module import path (Ray actors import
        # this module transitively and would pay ~9s for the deepspeed import)
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

        checkpoint_path = os.path.join(checkpoint_dir, tag)
        output_path = os.path.join(checkpoint_path, f"{tag}_universal")
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir=checkpoint_path, output_dir=output_path)


# --------------------------------------------------------------------------- #
# Torch-native (DCP) checkpointing -- used by TorchNativeTrainer
# --------------------------------------------------------------------------- #

# canonical state keys inside a DCP checkpoint
MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
TRAIN_STATE = "train_state"


class ModelWrapper(Stateful):
    """Flat FQN-keyed model state for DCP.

    FSDP2-sharded modules natively return sharded ``DTensor``s from
    ``state_dict()``, which DCP saves/loads reshard-safely. Model keys are
    flattened to the top level of the checkpoint (DCP-canonical layout), so
    after ``dcp.load`` the caller must route them back via ``load_state_dict``.
    """

    def __init__(self, model_parts: List[torch.nn.Module]):
        self.model_parts = model_parts

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for m in self.model_parts for k, v in m.state_dict().items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for m in self.model_parts:
            m.load_state_dict(state_dict, strict=False)


class DCPCheckpointManager:
    """torch.distributed.checkpoint manager for the torch-native trainer.

    Layout: ``<save_checkpointdir>/step-<N>/`` DCP shards + a
    ``checkpoint_meta.json`` sidecar (same schema as the DeepSpeed manager's).

    Hook contract (PeriodicCheckpointer/BestCheckpointer TORCHTITAN branches):
    ``save(curr_step, last_step=False)`` -- gated internally on ``save_period``
    -- and ``close()``. Resume goes through ``load()`` (called by
    ``helpers.resume_model_state``), which restores every registered Stateful
    in place and returns ``(loaded_step, checkpoint_meta)``.

    Unlike the DeepSpeed manager, checkpoints are reshard-safe: a run may
    resume on a different world size (``get_optimizer_state_dict`` flattens
    optimizer state to FQN keys; model state is sharded DTensors).
    """

    def __init__(
        self,
        model_parts: Optional[List[torch.nn.Module]] = None,
        optimizers: Optional[Stateful] = None,
        lr_schedulers: Optional[Stateful] = None,
        states: Optional[Dict[str, Any]] = None,
        save_checkpointdir: Union[str, Path] = None,
        save_period: int = 1,
        keep_latest_k: int = 0,
        resume_checkpointdir: Optional[Union[str, Path]] = None,
        pretrained_checkpointdir: Optional[Union[str, Path]] = None,
        load_optimizer: Optional[bool] = None,
        # eval-time trainers (TestTrainer/Inferencer) instantiate checkpoint
        # managers with a single `model=` kwarg -- accept it as an alias
        model: Optional[torch.nn.Module] = None,
    ):
        if resume_checkpointdir is not None and pretrained_checkpointdir is not None:
            # survives python -O (mirrors the DeepSpeed manager's guard)
            raise ValueError(
                "Cannot specify both `resume_checkpointdir` and `pretrained_checkpointdir`. "
                "Please choose one of them or neither."
            )
        if model_parts is None:
            if model is None:
                raise ValueError("DCPCheckpointManager requires model_parts (or model=).")
            model_parts = [model]

        self.model_parts = model_parts
        self.states: Dict[str, Any] = {MODEL: ModelWrapper(model_parts)}
        if optimizers is not None:
            self.states[OPTIMIZER] = optimizers
        if lr_schedulers is not None:
            self.states[LR_SCHEDULER] = lr_schedulers
        if states:
            overlap = set(states) & set(self.states)
            if overlap:
                raise ValueError(f"Duplicate checkpoint state keys: {sorted(overlap)}")
            self.states.update(states)

        self.save_checkpointdir = Path(save_checkpointdir)
        self.save_period = int(save_period)
        self.keep_latest_k = int(keep_latest_k)
        self.resume_checkpointdir = (
            Path(resume_checkpointdir) if resume_checkpointdir else None
        )
        self.pretrained_checkpointdir = (
            Path(pretrained_checkpointdir) if pretrained_checkpointdir else None
        )
        # default mirrors the DeepSpeed manager: load optimizer state only when resuming
        self.load_optimizer = (
            load_optimizer
            if load_optimizer is not None
            else self.resume_checkpointdir is not None
        )
        self._last_saved_step: Optional[int] = None

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _checkpoint_steps(ckpt_dir: Path) -> List[int]:
        """Completed checkpoint steps in a directory (require DCP .metadata)."""
        steps = []
        for p in ckpt_dir.glob("step-*"):
            if (p / ".metadata").exists():
                try:
                    steps.append(int(p.name.split("-")[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(steps)

    def _checkpoint_id(self, step: int, folder: Optional[Path] = None) -> str:
        return str((folder or self.save_checkpointdir) / f"step-{step}")

    def _flattened_sd(self, include_optimizer: bool = True) -> Dict[str, Any]:
        """Model keys flattened to top level; everything else nested Stateful."""
        sd = {
            k: v
            for k, v in self.states.items()
            if k != MODEL and (include_optimizer or k != OPTIMIZER)
        }
        sd.update(self.states[MODEL].state_dict())
        return sd

    def _build_metadata(self, curr_step: int) -> Dict[str, Any]:
        trainer = self.states.get(TRAIN_STATE, None)
        if trainer is None:
            return default_metadata(reason="no train_state registered")
        return build_metadata(
            model=self.model_parts[0],
            cfg=getattr(trainer, "pristine_cfg", getattr(trainer, "cfg", {})),
            epoch=getattr(trainer, "_epoch", None),
            iter_=curr_step,
            best_loss=getattr(trainer, "best_metric", None),
            trainer_state=trainer.state_dict(),
        )

    def _should_save(self, curr_step: int, last_step: bool) -> bool:
        if curr_step == self._last_saved_step:
            return False  # e.g. after_epoch + after_train both fire at the same step
        if last_step:
            return True
        return self.save_period > 0 and curr_step % self.save_period == 0

    def _purge_stale(self) -> None:
        if self.keep_latest_k <= 0 or not is_main_process():
            return
        steps = self._checkpoint_steps(self.save_checkpointdir)
        for step in steps[: -self.keep_latest_k]:
            shutil.rmtree(self._checkpoint_id(step), ignore_errors=True)
            logger.info(f"[DCPCheckpointManager] purged stale checkpoint step-{step}")

    # ---------------------------------------------------------------- save --
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Collective. Saves when ``curr_step`` hits the period (or last_step)."""
        if not self._should_save(curr_step, last_step):
            return False
        self.save_checkpointdir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = self._checkpoint_id(curr_step)
        dcp.save(self._flattened_sd(), checkpoint_id=checkpoint_id)
        if is_main_process():
            write_metadata_json(
                Path(checkpoint_id) / "checkpoint_meta.json",
                self._build_metadata(curr_step),
            )
        self._last_saved_step = curr_step
        self._purge_stale()
        barrier()
        logger.info(f"[DCPCheckpointManager] saved checkpoint at step {curr_step}")
        return True

    # ---------------------------------------------------------------- load --
    def load(self, step: Optional[int] = None):
        """Restore states in place. Returns ``(loaded_step, checkpoint_meta)``.

        - ``resume_checkpointdir`` -> full restore (model/optim/schedulers/
          train_state; optimizer skipped when ``load_optimizer=False``)
        - ``pretrained_checkpointdir`` -> model weights only
        - neither -> ``(None, None)``
        """
        if self.resume_checkpointdir is not None:
            return self._load(
                self.resume_checkpointdir,
                step=step,
                model_only=False,
                include_optimizer=self.load_optimizer,
            )
        if self.pretrained_checkpointdir is not None:
            return self._load(self.pretrained_checkpointdir, step=step, model_only=True)
        return None, None

    def load_for_eval(self, job_type: str):
        """Model-weights-only load for TestTrainer / Inferencer."""
        if self.resume_checkpointdir is not None:
            raise ValueError(
                f"job_type={job_type!r} loads model weights only -- use "
                "`pretrained_checkpointdir`, not `resume_checkpointdir`."
            )
        if self.pretrained_checkpointdir is None:
            raise ValueError("load_for_eval requires `pretrained_checkpointdir`.")
        # return the sidecar dict only, matching CheckpointManager.load_for_eval
        _, checkpoint_meta = self._load(
            self.pretrained_checkpointdir, step=None, model_only=True
        )
        return checkpoint_meta

    def _load(
        self,
        ckpt_dir: Path,
        step: Optional[int],
        model_only: bool,
        include_optimizer: bool = True,
    ):
        steps = self._checkpoint_steps(ckpt_dir)
        if not steps:
            return None, None
        if step is None:
            step = steps[-1]
        elif step not in steps:
            raise ValueError(
                f"Requested checkpoint step {step} not found in {ckpt_dir} "
                f"(available: {steps})."
            )
        checkpoint_id = self._checkpoint_id(step, folder=ckpt_dir)

        if model_only:
            sd = self.states[MODEL].state_dict()
        else:
            sd = self._flattened_sd(include_optimizer=include_optimizer)
        dcp.load(sd, checkpoint_id=checkpoint_id)
        # DCP cannot route the flattened model keys back through a Stateful --
        # route them manually (torchtitan does the same).
        self.states[MODEL].load_state_dict(sd)

        meta = read_metadata_json(
            Path(checkpoint_id) / "checkpoint_meta.json", allow_missing=True
        )
        logger.info(
            f"[DCPCheckpointManager] loaded checkpoint step-{step} from {ckpt_dir} "
            f"(model_only={model_only})"
        )
        return step, meta

    def close(self) -> None:
        # slot for async_save draining if/when async checkpointing lands
        pass
