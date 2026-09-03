"""`_build_dataloader_config`: the per-epoch rebuild kwargs must mirror the initial
`get_dataloader_ray` call."""

import inspect

from omegaconf import OmegaConf

from cell_observatory_platform.data.dataloaders import _build_dataloader_config
from cell_observatory_platform.data.datasets.pretrain_dataset_ray import get_dataloader_ray


def test_build_dataloader_config_carries_selected_channel_localizations():
    """The channel selection must survive into the rebuild dict (as a list): the
    per-epoch rebuild consumes this dict, so dropping the key disables selection
    for all of training."""
    cfg = OmegaConf.create(
        {
            "clusters": {"batch_size_per_gpu": 2},
            "datasets": {"last_batch_policy": "drop"},
        }
    )
    dl_cfg = _build_dataloader_config(
        config=cfg,
        collate_fn=None,
        sample_store_desc=None,
        dp_degree=None,
        dp_rank=None,
        selected_channel_localizations=("membrane", "cytosol"),
    )
    assert dl_cfg["selected_channel_localizations"] == ["membrane", "cytosol"]
    assert dl_cfg["batch_size"] == 2


def test_build_dataloader_config_keys_are_accepted_by_get_dataloader_ray():
    """Every key in the rebuild dict is a parameter of get_dataloader_ray, so the
    per-epoch rebuild can neither crash on nor silently drop a kwarg."""
    cfg = OmegaConf.create(
        {
            "clusters": {"batch_size_per_gpu": 1},
            "datasets": {"last_batch_policy": "drop"},
        }
    )
    dl_cfg = _build_dataloader_config(
        config=cfg,
        collate_fn=None,
        sample_store_desc=None,
        dp_degree=None,
        dp_rank=None,
        selected_channel_localizations=None,
    )
    accepted = set(inspect.signature(get_dataloader_ray).parameters)
    assert set(dl_cfg) <= accepted, (
        f"dataloader_config keys {set(dl_cfg) - accepted} not accepted by "
        f"get_dataloader_ray -- the per-epoch rebuild would crash or drop them"
    )


def test_build_dataloader_config_passes_handles_through_unchanged():
    """Config, collate_fn and the sample-store descriptor are passed by identity;
    scalar fields are copied verbatim and None stays None (not [])."""
    cfg = OmegaConf.create({"clusters": {"batch_size_per_gpu": 4}, "datasets": {"last_batch_policy": "keep"}})
    collate, desc = object(), object()
    dl = _build_dataloader_config(config=cfg, collate_fn=collate, sample_store_desc=desc,
                                  dp_degree=8, dp_rank=3, selected_channel_localizations=None)
    assert dl["cfg"] is cfg and dl["collate_fn"] is collate and dl["sample_store_desc"] is desc
    assert (dl["dp_degree"], dl["dp_rank"], dl["last_batch_policy"], dl["batch_size"]) == (8, 3, "keep", 4)
    assert dl["selected_channel_localizations"] is None        # None must not become []
