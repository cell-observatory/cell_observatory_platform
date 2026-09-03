"""Assert every config kwarg that gets SPLATTED into a callable matches its signature.

Two call sites splat whole config nodes, and they are exactly the two that drifted:

  * ``training/loops.py`` builds the inferencer via ``REGISTRY.build`` ->
    ``instantiate_as`` (utils/config.py), which pops only ``name``/``BUILD`` -- every
    remaining key in ``inference.inferencer_worker`` becomes a constructor kwarg.
  * ``inference/visualizer.py`` submits each ``handler_configs.<handler>`` block as
    ``**kwargs`` to the handler; four handlers splat that on into a plotter.

The ``save_worker`` / ``viz_worker`` blocks are passed as explicit named kwargs instead,
so a stray key there is silently IGNORED rather than a TypeError -- hence the inverse
assertion for those.

Configs are fragments full of interpolations (``${storage_dtype}``,
``${datasets.input_shape}``) that only resolve inside a full experiment. Key sets are
read without resolving; the few root-level dtype leaves the save-handler derivation
needs are resolved against a stub root.
"""

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cell_observatory_platform.inference import utils as inference_utils
from cell_observatory_platform.inference import visualizer as viz_module
from cell_observatory_platform.inference.inferencer import InferencerWorker
from cell_observatory_platform.inference.saver import _derive_save_handler
from cell_observatory_platform.utils.registry import REGISTRY

# Populate REGISTRY (walk-imports every component root).
import cell_observatory_platform.utils._register  # noqa: F401


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "inference"

# Supplied by training/loops.py at build time, so legitimately absent from the config.
_BUILD_TIME = {
    "model", "buffer_manager", "save_worker", "viz_worker",
    "model_name", "timepoint_idxs_for_save",
}
# utils/config.py _SELECTORS, stripped before the splat.
_SELECTORS = {"name", "BUILD", "_target_"}

# Blocks loops.py hand-lists. Keys outside these sets are never read.
_NAMED = {
    "save_worker": {"max_workers", "save_mode", "chunk_spatial_shape", "shard_spatial_shape"},
    "viz_worker": {"output_dir", "handler_configs", "max_workers"},
}

# handler -> (callable whose signature a handler_configs block must satisfy, names the
# handler binds itself). Four handlers splat **kwargs into a plotter; bbox_overlay
# consumes explicit kwargs and SWALLOWS the rest in **kwargs, so a stray key there is
# silently ignored -- the handler signature is the contract, with var_kw NOT honoured.
_VIZ = {
    "semantic_map": (inference_utils.save_semantic_predictions, {"name", "preds", "image", "targets", "save_dir"}),
    "instance_overlay": (inference_utils.save_instance_predictions,
                         {"save_dir", "identifier", "image", "preds", "targets", "region", "kinds"}),
    "save_predictions": (inference_utils.save_prediction_plots, {"name", "predictions", "save_dir"}),
    "feature_viz": (inference_utils.save_feature_visualizations, {"name", "predictions", "save_dir"}),
    "bbox_overlay": (viz_module.bbox_overlay_handler, {"record", "save_dir", "global_rank"}),
}
# Root-level interpolations the inference fragments reference (values as in
# configs/experiments/abc/base_experiments/*_inference.yaml).
_DTYPE_STUBS = {"dataset_dtype": "float16", "storage_dtype": "uint16"}

_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _keys(node) -> set:
    return set(OmegaConf.to_container(node, resolve=False).keys())


def _sig(fn):
    params = inspect.signature(fn).parameters
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    names = {n for n, p in params.items() if p.kind in kinds} - {"self"}
    required = {
        n for n, p in params.items()
        if p.default is inspect.Parameter.empty and p.kind in kinds
    } - {"self"}
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return names, required, var_kw


def _load(path: Path):
    root = OmegaConf.merge(OmegaConf.create(_DTYPE_STUBS), OmegaConf.load(path))
    # Neither surviving config carries a package directive, so composing by name would
    # nest under `inference`; tolerate both shapes. The child keeps its parent, so
    # ${storage_dtype} still resolves.
    return root.get("inference", root)


def test_configs_present():
    """Guard against the glob silently matching nothing (which would make every
    parametrized test below vacuous)."""
    assert _CONFIGS, f"no inference configs found under {CONFIG_DIR}"


def test_every_registered_viz_handler_has_a_contract_entry():
    """A newly registered viz handler must declare its kwarg contract here, or the
    signature check below would KeyError on the first config that uses it."""
    assert set(REGISTRY.names("viz_handler")) == set(_VIZ)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_inferencer_worker_kwargs_match_signature(path):
    """Every key under `inferencer_worker` reaches InferencerWorker.__init__ verbatim."""
    cfg = _load(path)
    names, required, var_kw = _sig(InferencerWorker.__init__)
    keys = _keys(cfg.inferencer_worker) - _SELECTORS

    if not var_kw:
        unknown = keys - names
        assert not unknown, (
            f"{path.name}: inferencer_worker declares kwargs InferencerWorker does not "
            f"accept: {sorted(unknown)}"
        )
    missing = required - keys - _BUILD_TIME
    assert not missing, (
        f"{path.name}: inferencer_worker is missing required kwargs: {sorted(missing)}"
    )


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_named_kwarg_blocks_are_actually_read(path):
    """save_worker / viz_worker keys are hand-forwarded; a stray key is silently dropped."""
    cfg = _load(path)
    for block, forwarded in _NAMED.items():
        if block not in cfg:
            continue
        extra = _keys(cfg[block]) - forwarded
        assert not extra, (
            f"{path.name}: {block} declares keys training/loops.py never forwards, so they "
            f"have no effect: {sorted(extra)}"
        )


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_viz_handler_kwargs_match_signature(path):
    """Each handler_configs block is splatted into its handler / plotter as **kwargs."""
    cfg = _load(path)
    handler_configs = OmegaConf.select(cfg, "viz_worker.handler_configs")
    if handler_configs is None:
        pytest.skip("no viz_worker.handler_configs block")

    for handler_name, kwargs_node in handler_configs.items():
        assert REGISTRY.has("viz_handler", handler_name), (
            f"{path.name}: '{handler_name}' is not a registered viz_handler"
        )
        target, bound = _VIZ[handler_name]
        names, required, _ = _sig(target)
        keys = _keys(kwargs_node)

        unknown = keys - names
        assert not unknown, (
            f"{path.name}: handler '{handler_name}' passes kwargs {target.__name__} does not "
            f"accept: {sorted(unknown)}"
        )
        collide = keys & bound
        assert not collide, (
            f"{path.name}: handler '{handler_name}' re-passes handler-bound kwargs "
            f"(duplicate-kwarg TypeError): {sorted(collide)}"
        )
        missing = required - keys - bound
        assert not missing, (
            f"{path.name}: handler '{handler_name}' is missing required kwargs for "
            f"{target.__name__}: {sorted(missing)}"
        )


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_save_tensors_entries_resolve_to_a_handler(path):
    """`save_tensors` is read key-by-key rather than splatted, so the signature checks
    above cannot see it. This guards the dispatch that silently no-op'd: every entry
    must carry the schema keys AND derive to a registered save_handler, with its
    dtype resolved against the real root-level interpolation targets."""
    cfg = _load(path)
    base = "inferencer_worker.outputs_metadata.save_tensors"
    save_tensors = OmegaConf.select(cfg, base)
    if save_tensors is None:
        pytest.skip("no save_tensors block")

    registered = set(REGISTRY.names("save_handler"))
    for tensor in save_tensors:
        meta = OmegaConf.to_container(save_tensors[tensor], resolve=False)
        required = {"name", "dtype", "annotation_type", "data_format"}
        assert required <= set(meta), (
            f"{path.name}: save_tensors.{tensor} is missing schema keys "
            f"{sorted(required - set(meta))}"
        )
        meta["dtype"] = str(OmegaConf.select(cfg, f"{base}.{tensor}.dtype"))   # resolve ONLY this leaf
        assert not meta["dtype"].startswith("${"), (
            f"{path.name}: save_tensors.{tensor} has an unstubbed interpolation {meta['dtype']}"
        )
        handler = _derive_save_handler(meta)      # raises on dense non-mask with a non-float dtype
        assert handler in registered, (
            f"{path.name}: save_tensors.{tensor} derives to unregistered handler {handler!r}"
        )
