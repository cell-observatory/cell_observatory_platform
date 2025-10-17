import os
import sys
import logging
import subprocess
from pathlib import Path 

from collections import OrderedDict
from typing import Dict, Optional, Union, Literal, List

import torch 

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from data.data_types import TORCH_DTYPES
from utils.context import is_main_process, barrier

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self,
                 model: torch.nn.Module, 
                 zero_stage: int,
                 load_universal_checkpoint: bool,
                 save_checkpointdir: Union[str, Path],
                 resume_checkpointdir: Optional[Union[str, Path]] = None,
                 pretrained_checkpointdir: Optional[Union[str, Path]] = None,
                 engine: Literal["deepspeed"] = "deepspeed", 
                 checkpoint_tag: str = "best_model",
                 load_dtype: Optional[Literal["fp16", "bf16"]] = None,
                 state_dict_filter: Optional[List[str]] = None,
                 freeze_modules: Optional[Union[str, List[str]]] = None,
                 activation_checkpoint_modules: Optional[Union[str, List[str]]] = None
    ):
        self.model = model
        self.engine = engine
        self.zero_stage = zero_stage
        self.load_dtype = load_dtype
        self.checkpoint_tag = checkpoint_tag
        self.load_universal_checkpoint = load_universal_checkpoint

        assert not (resume_checkpointdir is not None and \
            pretrained_checkpointdir is not None), \
            "Cannot specify both `resume_checkpointdir` and `pretrained_checkpointdir`. " \
            "Please choose one of them or neither." 
        
        self.resume_checkpointdir = resume_checkpointdir
        self.pretrained_checkpointdir = pretrained_checkpointdir

        if resume_checkpointdir is not None:
            self.load_checkpointdir = Path(resume_checkpointdir)
        elif pretrained_checkpointdir is not None:
            self.load_checkpointdir = Path(pretrained_checkpointdir)
        else:
            self.load_checkpointdir = None
        
        self.save_checkpointdir = Path(save_checkpointdir) \
            if isinstance(save_checkpointdir, str) else save_checkpointdir

        self.freeze_modules = freeze_modules
        self.activation_checkpoint_modules = activation_checkpoint_modules
        self.state_dict_filter = list(state_dict_filter) if state_dict_filter \
            is not None else None

    def save(self, 
             prefix: str, 
             save_epoch: int = None,
             save_step: int = None, 
             save_best_loss: Optional[float] = None,
    ):
        self.save_checkpointdir.mkdir(parents=True, exist_ok=True)
        if self.engine == "deepspeed":
            client_state = {
                "epoch": save_epoch,
                "iter": save_step,
                "best_loss": save_best_loss
            }
            self.model.save_checkpoint(self.save_checkpointdir, client_state=client_state, tag=prefix)
        else:
            raise NotImplementedError("Saving checkpoints for " \
                "other engines not implemented yet.")

    def load(self):  
        assert self.load_checkpointdir is not None, \
            "No checkpoint directory specified for loading. " \
            "Please set `resume_checkpointdir` or `pretrained_checkpointdir`."
        
        ckpt_zero_stage = self._get_zero_stage(os.path.join(self.load_checkpointdir, 
                                                    self.checkpoint_tag))

        if self.state_dict_filter is not None and self.engine == "deepspeed":
            custom_filter_fn = self._state_dict_filter_fn
        else:
            custom_filter_fn = None

        # we do not currently support loading ZeRO-0 checkpoints
        # into ZeRO-1/2/3 models
        if (self.zero_stage == 0 and ckpt_zero_stage != 0) or \
            (self.zero_stage != 0 and ckpt_zero_stage == 0):
            raise ValueError(
                f"Cannot load a ZeRO-0 checkpoint into a \
                    ZeRO-{self.zero_stage} model. "
            )

        # if we are resuming from a checkpoint that has an identical zero stage 
        # and zero configuration, which is specified by setting 
        # `load_universal_checkpoint` to False and `resume_checkpointdir` to a valid path
        # we can load the checkpoint directly with the load_checkpoint API      
        elif self.resume_checkpointdir is not None and not self.load_universal_checkpoint:
            ckpt_path, client_state = self.model.load_checkpoint(self.load_checkpointdir, 
                                          self.checkpoint_tag, 
                                          custom_load_fn=custom_filter_fn)
        
        # if we are resuming from a checkpoint that has a different zero configuration
        # we first attempt to find an existing universal checkpoint to load.
        # if it does not exist, we convert the ZeRO-1/2/3 checkpoint to a universal
        # checkpoint and then load it. this is the new prefered way to ensure that 
        # the state of the model is loaded correctly when restarting a job with a different
        # zero stage or configuration. see: https://arxiv.org/pdf/2406.18820
        else:
            assert self.model.load_universal_checkpoint(), \
                "Model does not support loading universal checkpoints. " \
                "Please set `allow_universal_conversion` to True in " \
                "DeepSpeed Config."
            
            universal_checkpointdir = self.load_checkpointdir / f"{self.checkpoint_tag}_universal"

            if universal_checkpointdir.exists() and any(universal_checkpointdir.iterdir()):
                logger.info(f"Loading universal checkpoint from existing directory \
                        {self.load_checkpointdir / f'{self.checkpoint_tag}_universal'}")
                ckpt_path, client_state = self.model.load_checkpoint(
                    self.load_checkpointdir, 
                    tag=f"{self.checkpoint_tag}_universal",
                    custom_load_fn=custom_filter_fn
                )
            else:

                logger.info(f"Converting ZeRO-0 checkpoint to universal format: \
                    {universal_checkpointdir}"
                    "NOTE: this will create a new checkpoint \
                    with the tag `{self.checkpoint_tag}_universal` in the same directory."
                )

                if is_main_process():
                    self._convert_zero_checkpoint_to_universal(
                        src=self.load_checkpointdir / self.checkpoint_tag,
                        dst=self.load_checkpointdir / f"{self.checkpoint_tag}_universal"
                    )
                
                barrier()

                ckpt_path, client_state = self.model.load_checkpoint(self.load_checkpointdir, 
                                        tag=f"{self.checkpoint_tag}_universal",
                                        custom_load_fn=custom_filter_fn)

        # get target dtype if specified
        if self.load_dtype is not None:
            module = getattr(self.model, "module", self.model)
            module.to(TORCH_DTYPES[self.load_dtype].value)

        if self.activation_checkpoint_modules is not None \
            or self.freeze_modules is not None:
            # NOTE: assumes that the model has
            #      `activation_checkpoint` and `freeze` methods
            base = getattr(self.model, "module", self.model)

            if self.activation_checkpoint_modules:
                base.activation_checkpoint(base, self.activation_checkpoint_modules)

            if self.freeze_modules:
                base.freeze(self.freeze_modules)

        return ckpt_path, client_state
    
    # custom_load_fn from DeepSpeed accepts 
    # a state_dict src and loads it into dst
    def _state_dict_filter_fn(self,
                              src: Dict[str, torch.Tensor],
                              dst: torch.nn.Module) -> None:
        if self.state_dict_filter is None:
            dst.load_state_dict(src, strict=False)
            return

        filtered = {k: v for k, v in src.items()
                    if any(k.startswith(p) for p in self.state_dict_filter)}

        missing, unexpected = dst.load_state_dict(filtered, strict=False)

        if missing:
            logger.warning(f"[CheckpointManager] missing keys after filter: {missing}")
        if unexpected:
            logger.warning(f"[CheckpointManager] unexpected keys after filter: {unexpected}")

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

    def _prefix_aware_load_state_dict(self, 
                                      state_dict: Dict[str, torch.Tensor], 
                                      model: torch.nn.Module
    ):
        ckpt_has_module   = any(k.startswith("module.") for k in state_dict)
        model_expects_mod = any(k.startswith("module.") for k in model.state_dict())

        if ckpt_has_module and not model_expects_mod:
            # Loading a DDP checkpoint into a non‑DDP model
            state_dict = self._strip_prefix(state_dict, prefix="module.")
        elif not ckpt_has_module and model_expects_mod:
            # Loading a non‑DDP checkpoint into a DDP‑wrapped model
            state_dict = self._add_prefix(state_dict, prefix="module.")
        
        model.load_state_dict(state_dict)

    @staticmethod
    def _add_prefix(state_dict: Dict[str, torch.Tensor],
                    prefix: str = "module."
    ) -> Dict[str, torch.Tensor]:
        """
        If none of the keys start with `prefix`, add it to every key.
        Otherwise return the dict unchanged.
        """
        return OrderedDict(
            (k if k.startswith(prefix) else f"{prefix}{k}", v)
            for k, v in state_dict.items()
        )

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
                new_key = key[len(prefix):]
                new_state[new_key] = value
            return new_state
        else:
            return state_dict

    def _convert_zero_checkpoint_to_universal(self, 
                                              src: Path, 
                                              dst: Path):
        """
        Convert a DeepSpeed Zero Stage 3 checkpoint to a standard checkpoint
        that may be loaded more flexibly.
        """
        if not dst.exists():
            os.makedirs(dst)

        cmd = [
            sys.executable, "-m", "deepspeed.checkpoint.ds_to_universal",
            "--input_folder", str(src),
            "--output_folder", str(dst),
            "--inject_missing_state"
        ]

        subprocess.check_call(cmd)

    # useful utility function to convert a deepspeed 
    # zero stage 3 checkpoint to a standard checkpoint
    def convert_zero_checkpoint(checkpoint_dir: str, tag: str = "best_model"):
        checkpoint_path = os.path.join(checkpoint_dir, tag)
        output_path = os.path.join(checkpoint_path, f"{tag}_universal")
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir=checkpoint_path,
                                                           output_dir=output_path)