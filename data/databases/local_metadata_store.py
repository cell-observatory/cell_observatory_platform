from __future__ import annotations

import re
import json
import time
import logging

from enum import Enum
from hashlib import sha1
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc
from omegaconf import OmegaConf

from cell_observatory_platform.data.data_types import DataKind, kind_family
from cell_observatory_platform.data.datasets.utils import resolve_channel_indices
from cell_observatory_platform.data.databases.schema import (
    required_columns,
    validate_non_null,
    validate_projection,
)
from cell_observatory_platform.utils.context import barrier


logger = logging.getLogger(__name__)


AxisKey = Literal["T", "Z", "Y", "X", "C"]

# Per-channel catalog: one row per (roi_id, channel_idx) with SCALAR columns --
# the indexable form of the same information the training views expose as aligned
# arrays.

CHANNEL_CATALOG = "api.roi_channels"

# Axis -> the view column that pins it. C is dynamic on every view, so it never
# gets a shape predicate (the data_channel_count filters cover it).
_AXIS_COLUMN: dict[str, str] = {
    "T": "time_size", "Z": "z_size", "Y": "y_size", "X": "x_size",
}


def _normalize_channel_token(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def channel_role(channel_type: object, annotation_type: object) -> Optional[str]:
    """Concrete role for one channel, or ``None`` when it is signal (-> INPUT).

    We have :class:`DataKind`: annotation_type + '_masks'  -> instance_masks | semantic_masks
    """
    if _normalize_channel_token(channel_type) != "mask":
        return None
    nk = _normalize_channel_token(annotation_type)
    if not nk or nk == "none":
        raise ValueError("channel_type='mask' with no annotation_type")
    return f"{nk}_masks"


def is_object_role(role: object) -> bool:
    """True if ``role`` names a GT channel kind (an instance/semantic mask family).

    Sourced from :class:`DataKind` rather than a second hand-maintained set, so
    the two vocabularies cannot drift apart.

    An UNKNOWN token is False, not an error. ``partition_channels`` asks this of
    every channel that carries a role, and a role need not be a DataKind at all --
    a data channel may legitimately carry a biology token ("membrane"). Raising
    here would reject the common case. The dangerous case -- a role that matches a
    declared TARGET family but is not an object role, i.e. GT about to be handed
    to the model as input -- is caught by ``partition_channels`` itself, which is
    where the two pieces of information meet.
    """
    try:
        family = kind_family(_normalize_channel_token(role))
    except ValueError:
        return False
    return family in (DataKind.INSTANCE_MASKS, DataKind.SEMANTIC_MASKS)


class SampleType(str, Enum):
    CUBE = "cube"
    TILE = "tile"


@dataclass(frozen=True)
class ViewSpec:
    sample_type: SampleType
    schema: str
    view: str
    fixed_axes: tuple[AxisKey, ...]
    dynamic_axes: tuple[AxisKey, ...]
    order_by: tuple[str, ...]
    has_occupancy_metrics: bool

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.view}"

    @property
    def is_cube(self) -> bool:
        return self.sample_type is SampleType.CUBE


TRAINING_VIEWS: dict[SampleType, ViewSpec] = {
    SampleType.CUBE: ViewSpec(
        sample_type=SampleType.CUBE,
        schema="api",
        view="cube_training",
        # A cube is a fixed-size crop: T/Z/Y/X come from cube_shapes and are
        # filtered on, so only the channel count varies row to row.
        fixed_axes=("T", "Z", "Y", "X"),
        dynamic_axes=("C",),
        # The DB's natural key, in its own order.
        order_by=(
            "source_kind", "roi_id", "tile_id",
            "time_start", "z_start", "y_start", "x_start",
        ),
        has_occupancy_metrics=True,
    ),
    SampleType.TILE: ViewSpec(
        sample_type=SampleType.TILE,
        schema="api",
        # NOTE the plural: api.tiles_training, not api.tile_training.
        view="tiles_training",
        # Tiles are natively ragged in space. 
        fixed_axes=(),
        dynamic_axes=("T", "Z", "Y", "X", "C"),
        order_by=("source_kind", "roi_id", "tile_id", "time_start", "time_size"),
        has_occupancy_metrics=False,
    ),
}


@dataclass(frozen=True)
class QuerySpec:
    roi_ids: Optional[Sequence[int]] = None
    tile_list: Optional[Sequence[str]] = None
    timepoint_list: Optional[Sequence[int]] = None
    max_rows: Optional[int] = None
    cdf_threshold: Optional[float] = None
    cdf_target: Literal["80", "90", "95", "99"] = "90"
    cdf_threshold_channel_localizations: Optional[Sequence[str]] = None
    holdout_split: Optional[Literal["train", "test"]] = None
    synthetic_only: bool = False
    complete_tiles_only: bool = False
    data_channel_count: Optional[int] = None
    min_data_channel_count: Optional[int] = None
    max_data_channel_count: Optional[int] = None
    required_channel_key_patterns: Optional[dict[str, str]] = None
    required_channel_localizations: Optional[Sequence[str]] = None
    required_fluorophores: Optional[Sequence[str]] = None
    required_annotation_types: Optional[Sequence[str]] = None

    def validate(self) -> None:
        for name in ("data_channel_count", "min_data_channel_count", "max_data_channel_count"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be >= 0")
        if (
            self.min_data_channel_count is not None
            and self.max_data_channel_count is not None
            and int(self.min_data_channel_count) > int(self.max_data_channel_count)
        ):
            raise ValueError("min_data_channel_count must be <= max_data_channel_count")
        for nk in self.required_annotation_types or ():
            if _normalize_channel_token(nk) not in {"instance", "semantic"}:
                raise ValueError(
                    f"Unknown annotation_type {nk!r}; dry_lab.annotation_types is "
                    f"seeded with 'instance' and 'semantic'"
                )


@dataclass(frozen=True)
class StoreSpec:
    root_dir: str


@dataclass(frozen=True)
class ResolvedSource:
    view: ViewSpec
    requested_time_size: int
    requested_z_size: int
    requested_y_size: int
    requested_x_size: int
    with_targets: bool
    location_id: int
    location_name: str

    @property
    def shape_key(self) -> str:
        return (
            f"{self.requested_time_size}_{self.requested_z_size}"
            f"_{self.requested_y_size}_{self.requested_x_size}"
        )

    @property
    def cache_key(self) -> str:
        return f"{self.view.qualified_name}:{self.shape_key}:{self.location_name}"


@dataclass(frozen=True)
class TableStats:
    num_rows: int
    max_time_size: int
    max_z_size: int
    max_y_size: int
    max_x_size: int
    max_channel_size: int
    ordering_fingerprint: str
    sample_type: str
    fixed_axes: tuple[AxisKey, ...]
    dynamic_axes: tuple[AxisKey, ...]
    # Channels the loader will actually EMIT per sample under the configured
    # selection: matched data channels PLUS the always-retained mask channels.
    # 0 when no selection is configured (then max_channel_size is the answer).
    # See _selected_channel_size for why len(selection) is not this number.
    max_selected_channel_size: int = 0


@dataclass(frozen=True)
class SharedTableDescriptor:
    path: str
    num_rows: int
    source_key: str


@dataclass(frozen=True)
class MappedTableDescriptor:
    sample_table: SharedTableDescriptor
    stats: TableStats


def _safe_max(table: pa.Table, column_name: str, default: int = 0) -> int:
    if column_name not in table.column_names or table.num_rows == 0:
        return int(default)
    value = pc.max(table[column_name]).as_py()
    return int(default if value is None else value)


def _ordering_fingerprint(table: pa.Table, view: ViewSpec) -> str:
    """
    Fingerprint the row order, so a silently reordered source is detectable.
    """
    if table.num_rows == 0:
        return "empty"

    present = [name for name in view.order_by if name in table.column_names]
    if not present:
        return "no-order-columns"

    head = np.arange(min(table.num_rows, 512), dtype=np.int64)
    tail = np.arange(max(0, table.num_rows - 512), table.num_rows, dtype=np.int64)
    idx = np.unique(np.concatenate([head, tail]))
    subset = table.select(present).take(pa.array(idx, type=pa.int64()))
    payload = {
        "num_rows": int(table.num_rows),
        "keys": present,
        "samples": subset.to_pylist(),
    }
    return sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _selected_channel_size(
    table: pa.Table, selected_channel_localizations: Optional[Sequence[str]]
) -> int:
    """
    Max channels the loader will EMIT per sample under this selection.
    """
    if not selected_channel_localizations:
        return 0
    needed = {"roi_id", "channel_idx", "channel_type", "localization"}
    if not needed.issubset(table.column_names) or table.num_rows == 0:
        return 0

    # one row index per distinct roi_id
    first_seen: dict[int, int] = {}
    for position, roi_id in enumerate(table["roi_id"].to_pylist()):
        first_seen.setdefault(roi_id, position)

    sample = table.take(pa.array(sorted(first_seen.values()), type=pa.int64()))
    idxs = sample["channel_idx"].to_pylist()
    types = sample["channel_type"].to_pylist()
    locs = sample["localization"].to_pylist()

    largest = 0
    for channel_idx, channel_type, localization in zip(idxs, types, locs):
        selected = resolve_channel_indices(
            channel_idx, channel_type, localization, selected_channel_localizations
        )
        largest = max(largest, len(selected) if selected else len(channel_idx or ()))
    return largest


def build_table_stats(
    table: pa.Table,
    resolved: ResolvedSource,
    selected_channel_localizations: Optional[Sequence[str]] = None,
) -> TableStats:
    """Per-axis maxima that size the host buffer (see dataloaders._shape_from_stats).
    """
    return TableStats(
        num_rows=int(table.num_rows),
        max_time_size=_safe_max(table, "time_size", resolved.requested_time_size),
        max_z_size=_safe_max(table, "z_size", resolved.requested_z_size),
        max_y_size=_safe_max(table, "y_size", resolved.requested_y_size),
        max_x_size=_safe_max(table, "x_size", resolved.requested_x_size),
        max_channel_size=_safe_max(table, "channel_size", 0),
        ordering_fingerprint=_ordering_fingerprint(table, resolved.view),
        sample_type=resolved.view.sample_type.value,
        fixed_axes=resolved.view.fixed_axes,
        dynamic_axes=resolved.view.dynamic_axes,
        max_selected_channel_size=_selected_channel_size(
            table, selected_channel_localizations
        ),
    )


def _assert_requested_time_size(table: pa.Table, resolved: ResolvedSource) -> None:
    """Fail loudly when the view cannot serve the requested temporal extent.

    Cubes are already safe: T is a fixed axis, so _shape_clauses pins time_size
    and a mismatch is simply 0 rows. Tiles are not. Without this the failure is SILENT:
    _shape_from_stats keeps the config T=16 (max(16, 1)), the loader slices ONE
    frame, and the 4D pad branch broadcasts it --
    `dst[i][tt:T, ...] = dst[i][tt - 1, ...]` -- yielding 16 copies of frame 0.
    _assert_input_shape_spatial only checks Z/Y/X, so nothing else objects.
    """
    if "time_size" not in table.column_names:
        # Unreachable through create_or_attach: time_size is in
        # schema.LOADER_COLUMNS, so validate_projection has already run and would
        # have raised.
        raise ValueError(
            f"{resolved.view.qualified_name}: no time_size column, so the "
            f"temporal-extent guard cannot run. The projection is not the one "
            f"data/databases/schema.py declares -- call validate_projection first."
        )
    if table.num_rows == 0:
        # Legitimately reachable: any filter may select nothing. Nothing to
        # compare against, and an empty result is ALREADY reported loudly --
        # _log_filter_narrowing fires on num_rows == 0 and names the clause that
        # zeroed it, which for a cube T mismatch is the `time_size=N` step
        # (cubes pin T in the WHERE, so a bad request is 0 rows rather than
        # wrong-sized rows). Raising here would duplicate that with a worse
        # message and would fire on every legitimately-empty query.
        return
    got = _safe_max(table, "time_size", resolved.requested_time_size)
    want = int(resolved.requested_time_size)
    if got != want:
        raise ValueError(
            f"{resolved.view.qualified_name}: requested time_size={want} but the "
            f"view serves {got}. A 4D config would silently receive {got} real "
            f"frame(s) broadcast to {want}. Seed dry_lab.tile_time_windows "
            f"(tiles) or dry_lab.cube_shapes (cubes) with time_size={want}, or "
            f"set datasets.input_shape[0]={got}."
        )


class FilterBuilder:
    @staticmethod
    def _sql_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _sql_list(values: Sequence[object]) -> str:
        encoded = []
        for value in values:
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.isnumeric()):
                encoded.append(str(value))
            else:
                encoded.append("'" + str(value).replace("'", "''") + "'")
        return "(" + ",".join(encoded) + ")"

    # ------------------------------------------------------------------ #
    # channel predicates -- membership in the aligned arrays
    # ------------------------------------------------------------------ #

    @classmethod
    def _channel_semijoin(cls, alias: str, **predicates: object) -> str:
        """``EXISTS`` over :data:`CHANNEL_CATALOG` for one channel requirement.

        The ROI must have at least one channel row matching every predicate::

            EXISTS (SELECT 1 FROM api.roi_channels rc
                    WHERE rc.roi_id = s.roi_id AND rc.localization = 'membrane')

        Text comparisons are ``lower(rc.col) = <lowered>``, not a bare equality.
        The DB vocabulary is NOT uniformly cased: localization and annotation_type
        are lowercase, but fluorophore is mixed ("Electra2", "mTFP1", "mCitrine",
        "mKOK", "mstaygold").
        """
        conditions = [f"rc.roi_id = {alias}.roi_id"]
        for column, value in predicates.items():
            if isinstance(value, int):
                conditions.append(f"rc.{column} = {int(value)}")
            else:
                conditions.append(
                    f"lower(rc.{column}) = "
                    f"{cls._sql_string(_normalize_channel_token(value))}"
                )
        return (
            f"EXISTS (SELECT 1 FROM {CHANNEL_CATALOG} rc WHERE "
            + " AND ".join(conditions)
            + ")"
        )

    @classmethod
    def _localization_index_expr(cls, alias: str, localization: object) -> str:
        """1-based position of a localization in the aligned channel arrays.

        POSITIONAL only -- filtering goes through _channel_semijoin. This exists to
        index ``occupancy_cdf*``, which the schema pins as "Length C
        (= channel_size), aligned to channel_idx", mask slots JSON null.
        """
        token = cls._sql_string(_normalize_channel_token(localization))
        return f"array_position({alias}.localization, {token})"

    @staticmethod
    def _channel_cdf_expr(alias: str, index_expr: str, target: str) -> str:
        """
        ``occupancy_cdf<target>[idx]`` -- a plain array subscript.
        """
        return f"({alias}.occupancy_cdf{target})[{index_expr}]"

    # ------------------------------------------------------------------ #
    # labeled clause groups
    # ------------------------------------------------------------------ #

    @classmethod
    def _shape_clauses(cls, resolved: "ResolvedSource", alias: str) -> list[tuple[str, str]]:
        """Shape as a filter, driven by ``ViewSpec.fixed_axes``.
        """
        sizes = {
            "T": resolved.requested_time_size,
            "Z": resolved.requested_z_size,
            "Y": resolved.requested_y_size,
            "X": resolved.requested_x_size,
        }
        return [
            (f"{_AXIS_COLUMN[ax]}={int(sizes[ax])}",
             f"{alias}.{_AXIS_COLUMN[ax]} = {int(sizes[ax])}")
            for ax in ("T", "Z", "Y", "X")
            if ax in resolved.view.fixed_axes
        ]

    @classmethod
    def _labeled_non_channel_clauses(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> list[tuple[str, str]]:
        """Ordered ``(label, sql_clause)`` pairs for non-channel predicates.

        Labels describe the filter for the diagnostic; clauses are the SQL. Single
        source of truth for which filters get applied -- both
        ``build_non_channel_where_sql`` and ``build_filter_diagnostic_sql``
        consume this.
        """
        out: list[tuple[str, str]] = []

        out.extend(cls._shape_clauses(resolved, alias))

        if query.roi_ids:
            out.append((
                f"roi_ids={list(query.roi_ids)}",
                f"{alias}.roi_id IN {cls._sql_list(query.roi_ids)}",
            ))

        if query.tile_list:
            out.append((
                f"tile_list ({len(query.tile_list)} names)",
                f"{alias}.tile_name IN {cls._sql_list(query.tile_list)}",
            ))

        if query.timepoint_list:
            out.append((
                f"timepoint_list={list(query.timepoint_list)}",
                f"{alias}.time_start IN {cls._sql_list(query.timepoint_list)}",
            ))

        if query.holdout_split == "train":
            out.append(("holdout_split=train", f"COALESCE({alias}.is_test_split, false) = false"))
        elif query.holdout_split == "test":
            out.append(("holdout_split=test", f"COALESCE({alias}.is_test_split, false) = true"))

        if query.synthetic_only:
            out.append((
                "synthetic_only",
                f"COALESCE({alias}.is_synthetic, false) = true",
            ))

        if query.complete_tiles_only and resolved.view.is_cube:
            # is_complete / missing_timepoints are projected on cube_training only;
            # tiles_training filters overlapping windows out internally.
            out.append(("complete_tiles_only", f"COALESCE({alias}.is_complete, false) = true"))

        if resolved.with_targets:
            out.append(("has_annotations", f"{alias}.has_annotations"))

        # data_channel_count excludes mask channels.
        if query.data_channel_count is not None:
            out.append((
                f"data_channel_count={int(query.data_channel_count)}",
                f"COALESCE({alias}.data_channel_count, 0) = {int(query.data_channel_count)}",
            ))
        if query.min_data_channel_count is not None:
            out.append((
                f"min_data_channel_count={int(query.min_data_channel_count)}",
                f"COALESCE({alias}.data_channel_count, 0) >= {int(query.min_data_channel_count)}",
            ))
        if query.max_data_channel_count is not None:
            out.append((
                f"max_data_channel_count={int(query.max_data_channel_count)}",
                f"COALESCE({alias}.data_channel_count, 0) <= {int(query.max_data_channel_count)}",
            ))

        return out

    @classmethod
    def _labeled_channel_clauses(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> list[tuple[str, str]]:
        """Ordered ``(label, sql_clause)`` pairs for channel predicates.
        """
        out: list[tuple[str, str]] = []

        for key, value in (query.required_channel_key_patterns or {}).items():
            # Pins a localization to a SPECIFIC channel, which is a different
            # question from "is it present anywhere".
            # This matches on rc.channel_idx, NOT on a position in the aligned
            # arrays. 
            out.append((
                f"required_channel_key_pattern[{key}]={value!r}",
                cls._channel_semijoin(alias, channel_idx=int(key), localization=value),
            ))

        for localization in query.required_channel_localizations or ():
            out.append((
                f"required_channel_localization={localization!r}",
                cls._channel_semijoin(alias, localization=localization),
            ))

        for fluorophore in query.required_fluorophores or ():
            out.append((
                f"required_fluorophore={fluorophore!r}",
                cls._channel_semijoin(alias, fluorophore=fluorophore),
            ))

        for nk in query.required_annotation_types or ():
            out.append((
                f"required_annotation_type={nk!r}",
                cls._channel_semijoin(alias, annotation_type=nk),
            ))

        if query.cdf_threshold is not None:
            if not resolved.view.has_occupancy_metrics:
                raise NotImplementedError(
                    f"cdf_threshold is not supported for "
                    f"{resolved.view.qualified_name} (no occupancy_cdf* columns)"
                )
            if not query.cdf_threshold_channel_localizations:
                raise NotImplementedError(
                    "cdf_threshold_channel_localizations is required for cube metric filters"
                )
            threshold = float(query.cdf_threshold)
            for localization in query.cdf_threshold_channel_localizations:
                index_expr = cls._localization_index_expr(alias, localization)
                # "the ROI has this channel at all" is an indexed semi-join; the
                # positional subscript below then picks its slot in THIS row's
                # occupancy array, which no join can answer.
                out.append((
                    f"cdf_threshold channel resolved for {localization!r}",
                    cls._channel_semijoin(alias, localization=localization),
                ))
                out.append((
                    f"cdf_{query.cdf_target}>={threshold} for {localization!r}",
                    f"{cls._channel_cdf_expr(alias, index_expr, query.cdf_target)} >= {threshold}",
                ))

        return out

    @classmethod
    def build_labeled_clauses(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> list[tuple[str, str]]:
        """All cumulative ``(label, clause)`` pairs in apply order."""
        return (
            cls._labeled_non_channel_clauses(resolved, query, alias=alias)
            + cls._labeled_channel_clauses(resolved, query, alias=alias)
        )

    @classmethod
    def build_non_channel_where_sql(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> str:
        query.validate()
        labeled = cls._labeled_non_channel_clauses(resolved, query, alias=alias)
        if not labeled:
            return "1=1"
        return " AND ".join(clause for _, clause in labeled)

    @classmethod
    def build_channel_where_clauses(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> list[str]:
        return [clause for _, clause in cls._labeled_channel_clauses(resolved, query, alias=alias)]

    @classmethod
    def build_missing_channel_metadata_where_sql(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> Optional[str]:
        """Rows dropped purely because the ROI has no channel rows at all.
        """
        has_channel_requirements = bool(
            query.cdf_threshold_channel_localizations
            or query.required_channel_localizations
            or query.required_channel_key_patterns
            or query.required_fluorophores
            or query.required_annotation_types
        )
        if not has_channel_requirements:
            return None
        return (
            f"NOT EXISTS (SELECT 1 FROM {CHANNEL_CATALOG} rc "
            f"WHERE rc.roi_id = {alias}.roi_id)"
        )

    @classmethod
    def build_where_sql(
        cls, resolved: "ResolvedSource", query: QuerySpec, alias: str = "s"
    ) -> str:
        clauses = [cls.build_non_channel_where_sql(resolved, query, alias=alias)]
        clauses.extend(cls.build_channel_where_clauses(resolved, query, alias=alias))
        return " AND ".join(clauses)


class TableResolver:
    @staticmethod
    def _requested_shape(config) -> tuple[int, int, int, int]:
        layout = config.dataset_layout_order.upper()
        input_shape = tuple(config.datasets.input_shape)
        if layout == "ZYXC":
            return 1, input_shape[0], input_shape[1], input_shape[2]
        if layout == "TZYXC":
            return input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        raise NotImplementedError(f"Unsupported dataset_layout_order={layout!r}")

    @staticmethod
    def _sample_type(config) -> SampleType:
        value = getattr(config.datasets.databases, "sample_type", None)
        if value is None:
            raise ValueError("datasets.databases.sample_type is required")
        return SampleType(str(value).lower())

    @staticmethod
    def _sequence_or_none(value) -> Optional[list]:
        if value is None:
            return None
        values = list(value)
        return values or None

    @classmethod
    def _normalized_channel_localizations(cls, values) -> Optional[tuple[str, ...]]:
        raw_values = cls._sequence_or_none(values)
        if raw_values is None:
            return None
        normalized: list[str] = []
        for value in raw_values:
            token = _normalize_channel_token(value)
            if token and token not in normalized:
                normalized.append(token)
        return tuple(normalized) if normalized else None

    @staticmethod
    def _iter_localization_tokens(value: object) -> list[str]:
        token = _normalize_channel_token(value)
        return [token] if token else []

    @classmethod
    def _fetch_channel_vocabulary(cls, db_client) -> dict[str, set[str]]:
        """Localization / fluorophore tokens that actually appear on data channels.
        """
        table = db_client.execute_arrow(
            f"""
            SELECT DISTINCT localization, fluorophore
            FROM {CHANNEL_CATALOG}
            WHERE channel_type = 'data'
            """
        )
        vocabulary: dict[str, set[str]] = {"localization": set(), "fluorophore": set()}
        for column in vocabulary:
            if column not in table.column_names:
                continue
            for raw in table[column].to_pylist():
                if raw is None:
                    continue
                token = _normalize_channel_token(raw)
                if token:
                    vocabulary[column].add(token)
        return vocabulary

    @classmethod
    def _assert_known_tokens(
        cls, db_client, column: str, requested: Optional[Sequence[str]]
    ) -> None:
        """Reject a config naming a channel token the data does not have.

        Without this the same typo surfaces either as a silently empty dataset
        (a filter that matches nothing) or as a per-row ValueError raised deep
        inside a Ray loader actor, after the whole pipeline is already up. Here it
        is a startup error that lists what IS available.
        """
        if db_client is None or not requested:
            return
        known = cls._fetch_channel_vocabulary(db_client)[column]
        if not known:
            return
        unknown = sorted(
            {_normalize_channel_token(v) for v in requested} - known
        )
        if unknown:
            raise ValueError(
                f"Unknown channel {column}(s) {unknown}; data channels in "
                f"{CHANNEL_CATALOG} carry {sorted(known)}"
            )

    @classmethod
    def _loader_channel_config(
        cls,
        config,
        db_client=None,
    ) -> Optional[tuple[str, ...]]:
        raw_selected = getattr(config.datasets, "selected_channel_localizations", None)
        selected_localizations = cls._normalized_channel_localizations(raw_selected)

        cls._assert_known_tokens(db_client, "localization", selected_localizations)
        return selected_localizations

    @classmethod
    def _required_channel_localizations_from_config(
        cls,
        config,
        db_client=None,
    ) -> Optional[tuple[str, ...]]:
        required = cls._normalized_channel_localizations(
            getattr(config.datasets.databases, "required_channel_localizations", None)
        )
        selected_localizations = cls._loader_channel_config(config, db_client=db_client)
        if required is None and selected_localizations is not None:
            required = selected_localizations
        if required is not None and selected_localizations is not None:
            missing = sorted(set(selected_localizations) - set(required))
            if missing:
                raise ValueError(
                    "selected_channel_localizations must be a subset of "
                    f"datasets.databases.required_channel_localizations; got invalid values {missing}"
                )

        cls._assert_known_tokens(db_client, "localization", required)
        return required

    @classmethod
    def build_loader_channel_selection_from_config(
        cls,
        config,
        db_client=None,
    ) -> Optional[tuple[str, ...]]:
        return cls._loader_channel_config(config, db_client=db_client)

    @staticmethod
    def _to_tuple(val):
        return None if val is None else tuple(val)

    @staticmethod
    def _to_dict(val):
        return None if val is None else dict(val)

    @classmethod
    def build_query_spec_from_config(cls, config, *, db_client=None) -> QuerySpec:
        db = config.datasets.databases
        return QuerySpec(
            # renamed from roi_list: prepared_id and roi_id are different
            # sequences, so the old values are not portable (and the stale ones
            # land inside the new range). base_database.yaml drops `roi_list` so
            # a stale config fails at load rather than selecting other rows.
            roi_ids=cls._to_tuple(getattr(config.datasets, "roi_ids", None)),
            tile_list=cls._to_tuple(config.datasets.tile_list),
            timepoint_list=cls._to_tuple(config.datasets.timepoint_list),
            max_rows=config.datasets.get("max_rows", None),
            cdf_threshold=config.datasets.cdf_threshold,
            cdf_target=config.datasets.cdf_target,
            cdf_threshold_channel_localizations=cls._to_tuple(
                getattr(db, "cdf_threshold_channel_localizations", None)
            ),
            holdout_split=getattr(db, "holdout_split", None),
            synthetic_only=bool(config.datasets.synthetic_only),
            complete_tiles_only=bool(getattr(db, "complete_tiles_only", False)),
            # data_channel_count excludes the mask channel; `channel_count` and
            # friends are deleted from the config rather than renamed in place,
            # because they counted it.
            data_channel_count=getattr(db, "data_channel_count", None),
            min_data_channel_count=getattr(db, "min_data_channel_count", None),
            max_data_channel_count=getattr(db, "max_data_channel_count", None),
            required_channel_key_patterns=cls._to_dict(
                getattr(db, "required_channel_key_patterns", None)
            ),
            required_channel_localizations=cls._required_channel_localizations_from_config(
                config, db_client=db_client
            ),
            required_fluorophores=cls._required_fluorophores_from_config(config, db_client=db_client),
            required_annotation_types=cls._to_tuple(
                getattr(db, "required_annotation_types", None)
            ),
        )

    @classmethod
    def _required_fluorophores_from_config(
        cls, config, db_client=None
    ) -> Optional[tuple[str, ...]]:
        """Validated ``required_fluorophores``.

        Same startup check as the localizations, and it matters more here: the
        fluorophore vocabulary is mixed-case in the DB ("Electra2", "mTFP1"), so a
        plausible-looking config value is easy to get subtly wrong. The filter
        itself compares case-insensitively, so the only failure a typo produces is
        an empty dataset.
        """
        requested = cls._normalized_channel_localizations(
            getattr(config.datasets.databases, "required_fluorophores", None)
        )
        cls._assert_known_tokens(db_client, "fluorophore", requested)
        return requested

    @staticmethod
    def build_store_spec_from_config(config) -> StoreSpec:
        return StoreSpec(root_dir=str(config.datasets.databases.node_local_store_root))

    @staticmethod
    def _storage_location_name(config) -> str:
        """Config names a location; SQL filters by its id.
        """
        name = getattr(config.datasets.databases, "storage_location", None)
        if not name:
            raise ValueError(
                "datasets.databases.storage_location is required (a "
                "dry_lab.storage_locations name, e.g. 'abc' or 'synthetic'). "
            )
        return str(name)

    @classmethod
    def resolve_from_config(cls, config, *, db_client=None) -> ResolvedSource:
        requested_time, requested_z, requested_y, requested_x = cls._requested_shape(config)
        view = TRAINING_VIEWS[cls._sample_type(config)]

        location_name = cls._storage_location_name(config)
        location_id = -1
        if db_client is not None:
            db_client.assert_relation_exists(view.view, schema=view.schema)
            location_id = resolve_location(db_client, location_name)

        return ResolvedSource(
            view=view,
            requested_time_size=requested_time,
            requested_z_size=requested_z,
            requested_y_size=requested_y,
            requested_x_size=requested_x,
            # was: a different physical table. now: a WHERE clause.
            with_targets=bool(config.datasets.has_annotations),
            location_id=location_id,
            location_name=location_name,
        )


def fetch_storage_locations(db_client) -> dict[str, int]:
    """``{name -> id}`` for active storage locations."""
    table = db_client.execute_arrow(
        "SELECT id, name FROM dry_lab.storage_locations WHERE status = 'active'"
    )
    return {
        str(n): int(i)
        for i, n in zip(table["id"].to_pylist(), table["name"].to_pylist())
    }


def resolve_location(db_client, name: str) -> int:
    known = fetch_storage_locations(db_client)
    if name not in known:
        raise ValueError(
            f"Unknown storage location {name!r}; dry_lab.storage_locations has "
            f"{sorted(known)}"
        )
    return known[name]


OBJECT_TYPE_RELATIONS = ("api.object_types", "dry_lab.object_types")


def fetch_object_type_names(db_client) -> dict[int, str]:
    """``object_type_id -> nk``, for resolving semantic class names.

    The annotation leaves carry only the integer, deliberately -- the schema says
    "leaves stay object_type_id + object_subtype_ids; no object_type_nk" -- while
    ``semantic_classes`` names classes. One row per type, so a single small query
    at startup.
    """
    last_error: Optional[Exception] = None
    for relation in OBJECT_TYPE_RELATIONS:
        try:
            table = db_client.execute_arrow(f"SELECT id, nk FROM {relation}")
        except Exception as exc:  # relation absent in this export
            last_error = exc
            continue
        if relation != OBJECT_TYPE_RELATIONS[0]:
            logger.warning(
                "[metadata_store] %s not present; read the object-type catalog "
                "from %s instead. Semantic class names resolve either way, but "
                "the api view is the contract.",
                OBJECT_TYPE_RELATIONS[0],
                relation,
            )
        return {
            int(i): str(n)
            for i, n in zip(table["id"].to_pylist(), table["nk"].to_pylist())
        }
    raise RuntimeError(
        f"No object-type catalog found (tried {list(OBJECT_TYPE_RELATIONS)}); "
        f"semantic class names cannot be resolved. Last error: {last_error}"
    )


class SqlQueryPlanner:
    """Builds the one projection, plus the two diagnostics..
    """

    @staticmethod
    def _location_join(alias: str, location_id: int) -> str:
        """Resolve the absolute storage root server-side.
        """
        return (
            f"JOIN dry_lab.storage_locations loc "
            f"ON loc.id = {int(location_id)} "
            f"AND loc.id = ANY({alias}.present_locations)"
        )

    @classmethod
    def _projection(cls, resolved: ResolvedSource, alias: str) -> list[str]:
        cols = required_columns(
            with_targets=resolved.with_targets, cube=resolved.view.is_cube
        )
        return [
            ("loc.root_path AS storage_root" if c == "storage_root" else f"{alias}.{c}")
            for c in cols
            if c != "row_id"
        ]

    @classmethod
    def build_sql(cls, resolved: ResolvedSource, query: QuerySpec) -> str:
        alias, view = "s", resolved.view

        select_sql = ",\n                ".join(cls._projection(resolved, alias))
        where_sql = FilterBuilder.build_where_sql(resolved, query, alias=alias)
        order_sql = ", ".join(f"{alias}.{name}" for name in view.order_by)

        sql = f"""
            SELECT
                row_number() OVER (ORDER BY {order_sql}) - 1 AS row_id,
                {select_sql}
            FROM {view.qualified_name} {alias}
            {cls._location_join(alias, resolved.location_id)}
            WHERE {where_sql}
            ORDER BY {order_sql}
        """
        if query.max_rows is not None:
            sql += f"\nLIMIT {int(query.max_rows)}"
        return sql

    @classmethod
    def build_missing_channel_metadata_diagnostic_sql(
        cls, resolved: ResolvedSource, query: QuerySpec
    ) -> Optional[str]:
        alias = "s"
        where_sql = FilterBuilder.build_missing_channel_metadata_where_sql(
            resolved, query, alias=alias
        )
        if where_sql is None:
            return None

        return f"""
            SELECT
                count(*) AS dropped_row_count,
                count(DISTINCT {alias}.roi_id) AS dropped_roi_count
            FROM {resolved.view.qualified_name} {alias}
            WHERE {where_sql}
        """

    @classmethod
    def build_filter_diagnostic_sql(
        cls, resolved: ResolvedSource, query: QuerySpec
    ) -> tuple[str, list[str]]:
        """Build a single SQL that returns one COUNT per cumulative filter step.

        Returns ``(sql, labels)`` where the SQL produces rows of
        ``(step int, n_rows bigint)`` ordered by step. Index 0 is the unfiltered
        baseline; index i is "all clauses up to and including filter i".
        """
        alias = "s"
        relation = f"{resolved.view.qualified_name} {alias}"
        labeled = FilterBuilder.build_labeled_clauses(resolved, query, alias=alias)
        labels: list[str] = ["baseline (no filter)"]
        parts: list[str] = [
            f"SELECT 0 AS step, count(*)::bigint AS n_rows FROM {relation}"
        ]
        cumulative: list[str] = []
        for i, (label, clause) in enumerate(labeled, start=1):
            cumulative.append(clause)
            labels.append(label)
            where = " AND ".join(cumulative)
            parts.append(
                f"SELECT {i} AS step, count(*)::bigint AS n_rows "
                f"FROM {relation} WHERE {where}"
            )

        sql = "\nUNION ALL\n".join(parts) + "\nORDER BY step"
        return sql, labels


class MappedTable:
    def __init__(self, descriptor: MappedTableDescriptor) -> None:
        self.descriptor = descriptor
        self._source = None
        self._reader = None
        self._table: Optional[pa.Table] = None

    @staticmethod
    def _apply_server_path_override(table: pa.Table, override: Optional[str]) -> pa.Table:
        """Point every row at a different mount, for a cluster whose local path
        differs from the catalog's ``root_path``.
        """
        if override is None:
            return table
        if "storage_root" not in table.column_names:
            raise ValueError("storage_root column not found in table")
        col_idx = table.column_names.index("storage_root")
        new_col = pa.array([str(override)] * table.num_rows, type=pa.utf8())
        table = table.set_column(col_idx, "storage_root", new_col)
        logger.info(
            "[MappedTable] storage_root overridden to %s for %d rows",
            override,
            table.num_rows,
        )
        return table

    @staticmethod
    def _log_filter_narrowing(
        db_client,
        resolved: ResolvedSource,
        query: QuerySpec,
        triggered_by: str,
    ) -> None:
        """Run a per-filter COUNT breakdown and log it as a table.

        Each step shows the cumulative row count after applying the
        next filter clause. The first step where ``n_rows`` drops to 0
        is the filter responsible for the empty result.
        """
        diag_sql, labels = SqlQueryPlanner.build_filter_diagnostic_sql(resolved, query)
        if len(labels) <= 1:
            logger.warning(
                "[MappedTable] filter narrowing diagnostic skipped for %s "
                "(no filters configured; trigger=%s)",
                resolved.view.qualified_name,
                triggered_by,
            )
            return

        diag_t0 = time.perf_counter()
        try:
            diag_table = db_client.execute_arrow(diag_sql)
        except Exception as exc:
            logger.warning(
                "[MappedTable] filter narrowing diagnostic for %s failed: %s",
                resolved.view.qualified_name,
                exc,
            )
            return
        diag_elapsed = time.perf_counter() - diag_t0

        # Materialize results into a {step -> n_rows} map; the SQL
        # already orders by step but be defensive in case the backend
        # reorders.
        steps_col = diag_table["step"].to_pylist()
        rows_col = diag_table["n_rows"].to_pylist()
        step_to_rows = {int(s): int(r) for s, r in zip(steps_col, rows_col)}

        baseline = step_to_rows.get(0, 0)
        first_zero_step: Optional[int] = None
        line_width = max(len(label) for label in labels)
        lines = [
            f"[MappedTable] filter narrowing for {resolved.view.qualified_name} "
            f"(trigger={triggered_by}, diagnostic took {diag_elapsed:.2f}s):",
            f"  {'step':>4} | {'n_rows':>10} | {'delta':>10} | filter",
            f"  {'-' * 4}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * line_width}",
        ]
        prev_rows = baseline
        for step_idx, label in enumerate(labels):
            n_rows = step_to_rows.get(step_idx, 0)
            delta = n_rows - prev_rows if step_idx > 0 else 0
            delta_str = "" if step_idx == 0 else f"{delta:+d}"
            marker = ""
            if step_idx > 0 and prev_rows > 0 and n_rows == 0 and first_zero_step is None:
                first_zero_step = step_idx
                marker = "  <-- first 0-row step"
            lines.append(
                f"  {step_idx:>4} | {n_rows:>10} | {delta_str:>10} | {label}{marker}"
            )

        logger.warning("\n".join(lines))

        if first_zero_step is not None:
            logger.warning(
                "[MappedTable] %s: filter step %d (%r) reduced row count from "
                "%d to 0. Loosen or remove this filter to recover rows.",
                resolved.view.qualified_name,
                first_zero_step,
                labels[first_zero_step],
                step_to_rows.get(first_zero_step - 1, baseline),
            )

    @staticmethod
    def _root(node_id: str, resolved: ResolvedSource, query: QuerySpec, store: StoreSpec) -> Path:
        payload = {
            "node_id": node_id,
            "source_key": resolved.cache_key,
            "query": asdict(query),
        }
        key_hash = sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
        return Path(store.root_dir) / str(node_id) / key_hash

    @classmethod
    def create_or_attach(
        cls,
        db_client,
        resolved: ResolvedSource,
        query: QuerySpec,
        store: StoreSpec,
        node_id: str,
        local_rank: int,
        diagnostic_verbose: bool = False,
        server_path_override: Optional[str] = None,
        selected_channel_localizations: Optional[Sequence[str]] = None,
    ) -> "MappedTable":
        total_t0 = time.perf_counter()
        root = cls._root(node_id=node_id, resolved=resolved, query=query, store=store)
        root.mkdir(parents=True, exist_ok=True)

        descriptor_path = root / "descriptor.json"
        sample_path = root / "sample_table.arrow"

        if local_rank == 0:
            diagnostic_sql = SqlQueryPlanner.build_missing_channel_metadata_diagnostic_sql(resolved, query)
            if diagnostic_sql is not None:
                if diagnostic_verbose:
                    logger.warning(
                        "[MappedTable] running channel-mapping diagnostic query for %s; "
                        "this may be slow on large tables (100s+). "
                        "Set datasets.databases.diagnostic_verbose=false to skip.",
                        resolved.view.qualified_name,
                    )
                    diagnostic_t0 = time.perf_counter()
                    diagnostic_table = db_client.execute_arrow(diagnostic_sql)
                    logger.info(
                        "[MappedTable] diagnostic query for %s completed in %.2fs",
                        resolved.view.qualified_name,
                        time.perf_counter() - diagnostic_t0,
                    )
                    dropped_rows = int(diagnostic_table["dropped_row_count"][0].as_py() or 0)
                    dropped_rois = int(diagnostic_table["dropped_roi_count"][0].as_py() or 0)
                    if dropped_rows > 0:
                        logger.warning(
                            "[MappedTable] filtering out %s rows across %s roi_ids from %s because "
                            "the ROI has no channel rows (channel_idx IS NULL).",
                            dropped_rows,
                            dropped_rois,
                            resolved.view.qualified_name,
                        )
                else:
                    logger.info(
                        "[MappedTable] skipping channel-mapping diagnostic query for %s "
                        "(set datasets.databases.diagnostic_verbose=true to enable)",
                        resolved.view.qualified_name,
                    )
            sql = SqlQueryPlanner.build_sql(resolved, query)
            fetch_t0 = time.perf_counter()
            table = db_client.execute_arrow(sql)
            table = cls._apply_server_path_override(table, server_path_override)
            fetch_elapsed = time.perf_counter() - fetch_t0
            logger.info(
                "[MappedTable] fetched %s rows from %s in %.2fs",
                table.num_rows,
                resolved.view.qualified_name,
                fetch_elapsed,
            )

            # Auto-trigger the per-filter narrowing diagnostic on empty
            # results so the operator can see exactly which clause killed
            # the row count. Also runs when diagnostic_verbose is set.
            if table.num_rows == 0 or diagnostic_verbose:
                cls._log_filter_narrowing(
                    db_client=db_client,
                    resolved=resolved,
                    query=query,
                    triggered_by="empty result" if table.num_rows == 0 else "diagnostic_verbose=true",
                )

            # Boundary checks, rank 0 only, once per run. Everything downstream
            # (loader, collator, buffer sizing) treats this Arrow table as the
            # contract, so a violation caught here is a startup error instead of
            # a KeyError/TypeError several actors deep inside a Ray task.
            where = f"{resolved.view.qualified_name} projection"
            validate_projection(
                table,
                with_targets=resolved.with_targets,
                cube=resolved.view.is_cube,
                where=where,
            )
            validate_non_null(table, where=where)
            _assert_requested_time_size(table, resolved)

            write_t0 = time.perf_counter()
            with pa.OSFile(str(sample_path), "wb") as sink:
                with pa_ipc.new_file(sink, table.schema) as writer:
                    writer.write(table)
            logger.info(
                "[MappedTable] wrote sample table for %s to %s in %.2fs",
                resolved.view.qualified_name,
                sample_path,
                time.perf_counter() - write_t0,
            )

            descriptor = MappedTableDescriptor(
                sample_table=SharedTableDescriptor(
                    path=str(sample_path),
                    num_rows=int(table.num_rows),
                    source_key=resolved.cache_key,
                ),
                stats=build_table_stats(
                    table, resolved, selected_channel_localizations
                ),
            )
            descriptor_path.write_text(json.dumps(asdict(descriptor)))
            logger.info(
                "[MappedTable] descriptor for %s materialized in %.2fs",
                resolved.view.qualified_name,
                time.perf_counter() - total_t0,
            )

        barrier()

        raw = json.loads(descriptor_path.read_text())
        stats_raw = dict(raw["stats"])
        stats_raw["fixed_axes"] = tuple(stats_raw.get("fixed_axes", ()))
        stats_raw["dynamic_axes"] = tuple(stats_raw.get("dynamic_axes", ()))
        descriptor = MappedTableDescriptor(
            sample_table=SharedTableDescriptor(**raw["sample_table"]),
            stats=TableStats(**stats_raw),
        )
        logger.info(
            "[MappedTable] attached descriptor for %s from %s in %.2fs",
            resolved.view.qualified_name,
            descriptor_path,
            time.perf_counter() - total_t0,
        )
        return cls(descriptor)

    def table(self) -> pa.Table:
        if self._table is None:
            self._source = pa.memory_map(self.descriptor.sample_table.path, "r")
            self._reader = pa_ipc.RecordBatchFileReader(self._source)
            self._table = self._reader.read_all()
        return self._table

    def close(self) -> None:
        self._table = None
        self._reader = None
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
            self._source = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "MappedTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SampleIndexPlanner:
    @staticmethod
    def split_train_val(num_rows: int, split_fraction: Optional[float], seed: int) -> tuple[np.ndarray, np.ndarray]:
        if split_fraction is None or not (0.0 < float(split_fraction) < 1.0):
            return np.arange(num_rows, dtype=np.int64), np.asarray([], dtype=np.int64)

        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(num_rows)
        val_size = int(round(num_rows * float(split_fraction)))
        val_idx = np.sort(perm[:val_size].astype(np.int64))
        train_idx = np.sort(perm[val_size:].astype(np.int64))
        return train_idx, val_idx

    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = int(world_size)
        self.rank = int(rank)

    def plan_epoch(
        self,
        base_row_ids: np.ndarray,
        seed: Optional[int],
        shuffle: bool,
        batch_size: int,
        last_batch_policy: str,
    ) -> np.ndarray:
        row_ids = np.asarray(base_row_ids, dtype=np.int64).copy()

        if shuffle:
            if seed is None:
                raise ValueError("seed must be provided when shuffle=True")
            rng = np.random.default_rng(int(seed))
            rng.shuffle(row_ids)

        usable = (len(row_ids) // self.world_size) * self.world_size
        row_ids = row_ids[:usable]
        local = row_ids[self.rank :: self.world_size]

        if last_batch_policy == "drop":
            keep = (len(local) // batch_size) * batch_size
            local = local[:keep]

        return local
