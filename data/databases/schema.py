"""The Arrow column contract between the metadata store and the data pipeline.

Everything downstream of :class:`MappedTable` -- the loader's path join and
hypercube slice, the collator's target construction, ``_shape_from_stats``'
buffer sizing -- consumes an Arrow table with THESE columns. This module is the
only place that names them.

Column semantics come from the published ``api.cube_training`` /
``api.tiles_training`` JSON Schemas.
"""

from __future__ import annotations

from typing import Final

import pyarrow as pa


# Columns LoaderActor needs to locate and slice a hypercube.

# The channel arrays are positionally aligned (array_agg ... ORDER BY channel_idx):
#   channel_idx     the zarr C-axis index for this position -- may be SPARSE
#   channel_type    data | mask
#   localization    biology; JSON null on mask channels (roi_channels XOR check)
#   annotation_type instance | semantic; JSON null on data channels
# so position k of each describes channel `channel_idx[k]`.
LOADER_COLUMNS: Final[tuple[str, ...]] = (
    "storage_root", "tile_relative_path",
    "time_start", "time_size",
    "z_start", "y_start", "x_start",
    "z_size", "y_size", "x_size",
    "channel_size", "data_channel_count",
    "channel_idx", "channel_type", "localization", "annotation_type",
)

# Columns FinetuneCollatorActor needs to build per-instance targets and to read
# the semantic legend.
TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "annotation_count", "annotations_metadata",
)

# Row identity and ordering.
IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "row_id", "source_kind", "roi_id", "tile_id", "tile_name",
)

# Passthrough: not consumed by the loader, carried into metainfo for
# provenance / eval / logging. array_shape/dtype/codec let the loader assert the
# DB's view of the zarr matches what tensorstore actually opens.
PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "is_synthetic", "is_test_split", "array_shape", "dtype", "codec",
)

# Cube-only passthrough: tiles_training filters on missing_timepoints internally
# and does not project either column.
CUBE_ONLY_COLUMNS: Final[tuple[str, ...]] = (
    "is_complete", "missing_timepoints",
)

# Columns whose value must never be NULL once the row reaches the loader.
NON_NULL_COLUMNS: Final[tuple[str, ...]] = (
    "time_start", "time_size",
    "z_start", "y_start", "x_start",
    "z_size", "y_size", "x_size",
    "storage_root", "tile_relative_path",
)


def required_columns(*, with_targets: bool, cube: bool = True) -> tuple[str, ...]:
    """The projection this pipeline needs.

    ``with_targets`` follows ``datasets.has_annotations``; ``cube`` selects the
    cube-only passthrough (``is_complete`` / ``missing_timepoints``).
    """
    cols = IDENTITY_COLUMNS + LOADER_COLUMNS + PROVENANCE_COLUMNS
    if cube:
        cols = cols + CUBE_ONLY_COLUMNS
    return cols + TARGET_COLUMNS if with_targets else cols


def validate_projection(
    table: pa.Table, *, with_targets: bool, cube: bool = True, where: str
) -> None:
    """Fail at the DB boundary, loudly, naming both sides.

    Extra columns are fine (the view may grow); missing ones are not.
    """
    missing = [
        c for c in required_columns(with_targets=with_targets, cube=cube)
        if c not in table.column_names
    ]
    if missing:
        raise ValueError(
            f"{where}: projection is missing required columns {missing}. "
            f"Got {sorted(table.column_names)}. Either the view changed or "
            f"data/databases/schema.py is stale -- reconcile both, do not "
            f"paper over it with a stub column."
        )


def validate_non_null(table: pa.Table, *, where: str) -> None:
    """Reject NULLs in columns the loader dereferences unconditionally.
    """
    offenders = []
    for name in NON_NULL_COLUMNS:
        if name not in table.column_names:
            continue
        n_null = table[name].null_count
        if n_null:
            offenders.append(f"{name} ({n_null} null)")
    if offenders:
        raise ValueError(
            f"{where}: NULL in columns the loader dereferences: {offenders}. "
            f"Cube origins are NOT NULL in the schema, but tiles_training derives "
            f"z/y/x_size from nullable array_shape elements -- filter those rows "
            f"upstream rather than letting them reach _slice_hypercube."
        )
