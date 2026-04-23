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

from cell_observatory_platform.utils.context import barrier


logger = logging.getLogger(__name__)


LocationKey = Literal["is_available", "exists_prfs", "exists_aws", "exists_oak", "exists_abc"]
AxisKey = Literal["T", "Z", "Y", "X", "C"]

# TODO: migrate to a DB table
LOCATION_SERVER_PATHS: dict[str, str] = {
    "exists_abc": "/clusterfs/vast/Data/cell_observatory_training_datasets",
    "exists_prfs": "/groups/betzig/betziglab/CellObservatoryData",
}


def _normalize_channel_token(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _literal_channel_pattern(value: object) -> str:
    normalized = _normalize_channel_token(value)
    if not normalized:
        raise ValueError("Channel localization values must not be empty")
    return re.escape(normalized)


class SampleType(str, Enum):
    CUBE = "cube"
    TILE = "tile"


class TrainingTableKind(str, Enum):
    CUBE_WITH_ANNOTATIONS = "cube_with_annotations"
    CUBE_WITHOUT_ANNOTATIONS = "cube_without_annotations"
    TILE_WITH_ANNOTATIONS = "tile_with_annotations"
    TILE_WITHOUT_ANNOTATIONS = "tile_without_annotations"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    training_table_kind: TrainingTableKind
    table_name_template: str
    fixed_axes: tuple[AxisKey, ...]
    dynamic_axes: tuple[AxisKey, ...]
    time_size: Optional[int]
    z_size: Optional[int]
    y_size: Optional[int]
    x_size: Optional[int]
    order_by: tuple[str, ...]

    @property
    def sample_type(self) -> SampleType:
        if self.training_table_kind in {
            TrainingTableKind.CUBE_WITH_ANNOTATIONS,
            TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS,
        }:
            return SampleType.CUBE
        return SampleType.TILE

    @property
    def has_annotation_payload(self) -> bool:
        return self.training_table_kind in {
            TrainingTableKind.CUBE_WITH_ANNOTATIONS,
            TrainingTableKind.TILE_WITH_ANNOTATIONS,
        }

    @property
    def supports_metric_filters(self) -> bool:
        return self.training_table_kind in {
            TrainingTableKind.CUBE_WITH_ANNOTATIONS,
            TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS,
        }


def _general_source(
    key: str,
    training_table_kind: TrainingTableKind,
    table_name_template: str,
    fixed_axes: tuple[AxisKey, ...],
    dynamic_axes: tuple[AxisKey, ...],
    time_size: Optional[int],
    z_size: Optional[int],
    y_size: Optional[int],
    x_size: Optional[int],
    order_by: tuple[str, ...],
) -> SourceSpec:
    return SourceSpec(
        key=key,
        training_table_kind=training_table_kind,
        table_name_template=table_name_template,
        fixed_axes=fixed_axes,
        dynamic_axes=dynamic_axes,
        time_size=time_size,
        z_size=z_size,
        y_size=y_size,
        x_size=x_size,
        order_by=order_by,
    )


SOURCE_TABLES: dict[str, SourceSpec] = {
    "cube_with_annotations": _general_source(
        key="cube_with_annotations",
        training_table_kind=TrainingTableKind.CUBE_WITH_ANNOTATIONS,
        table_name_template="prepared_cube_annotation_agg_{time_size}_{z_size}_{y_size}_{x_size}",
        fixed_axes=("T", "Z", "Y", "X"),
        dynamic_axes=("C",),
        time_size=None,
        z_size=None,
        y_size=None,
        x_size=None,
        order_by=("prepared_id", "tile_name", "time_start", "z_start", "y_start", "x_start"),
    ),
    "cube_without_annotations": _general_source(
        key="cube_without_annotations",
        training_table_kind=TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS,
        table_name_template="prepared_cube_channel_agg_{time_size}_{z_size}_{y_size}_{x_size}",
        fixed_axes=("T", "Z", "Y", "X"),
        dynamic_axes=("C",),
        time_size=None,
        z_size=None,
        y_size=None,
        x_size=None,
        order_by=("prepared_id", "tile_name", "time_start", "z_start", "y_start", "x_start"),
    ),
    "tile_with_annotations": _general_source(
        key="tile_with_annotations",
        training_table_kind=TrainingTableKind.TILE_WITH_ANNOTATIONS,
        table_name_template="prepared_tile_annotation_agg_{time_size}",
        fixed_axes=(),
        dynamic_axes=("T", "Z", "Y", "X", "C"),
        time_size=None,
        z_size=None,
        y_size=None,
        x_size=None,
        order_by=("prepared_id", "tile_name", "time_start"),
    ),
    "tile_without_annotations": _general_source(
        key="tile_without_annotations",
        training_table_kind=TrainingTableKind.TILE_WITHOUT_ANNOTATIONS,
        table_name_template="prepared_tile_channel_agg_{time_size}",
        fixed_axes=(),
        dynamic_axes=("T", "Z", "Y", "X", "C"),
        time_size=None,
        z_size=None,
        y_size=None,
        x_size=None,
        order_by=("prepared_id", "tile_name", "time_start"),
    ),
}


@dataclass(frozen=True)
class QuerySpec:
    roi_list: Optional[Sequence[int]] = None
    tile_list: Optional[Sequence[str]] = None
    timepoint_list: Optional[Sequence[int]] = None
    max_rows: Optional[int] = None
    cdf_threshold: Optional[float] = None
    cdf_target: Literal["80", "90", "95", "99"] = "90"
    cdf_threshold_channel_localizations: Optional[Sequence[str]] = None
    required_locations: Optional[Sequence[LocationKey]] = None
    holdout_split: Optional[Literal["train", "test"]] = None
    synthetic_only: bool = False
    channel_count: Optional[int] = None
    min_channel_count: Optional[int] = None
    max_channel_count: Optional[int] = None
    required_channel_key_patterns: Optional[dict[str, str]] = None
    any_channel_patterns: Optional[Sequence[str]] = None
    all_channel_patterns: Optional[Sequence[str]] = None
    required_channel_localizations: Optional[Sequence[str]] = None

    def validate(self) -> None:
        for key in self.required_locations or ():
            if key not in {"is_available", "exists_prfs", "exists_aws", "exists_oak", "exists_abc"}:
                raise ValueError(f"Unknown location key {key!r}")
        if self.channel_count is not None and int(self.channel_count) < 0:
            raise ValueError("channel_count must be >= 0")
        if self.min_channel_count is not None and int(self.min_channel_count) < 0:
            raise ValueError("min_channel_count must be >= 0")
        if self.max_channel_count is not None and int(self.max_channel_count) < 0:
            raise ValueError("max_channel_count must be >= 0")
        if (
            self.min_channel_count is not None
            and self.max_channel_count is not None
            and int(self.min_channel_count) > int(self.max_channel_count)
        ):
            raise ValueError("min_channel_count must be <= max_channel_count")


@dataclass(frozen=True)
class StoreSpec:
    root_dir: str


@dataclass(frozen=True)
class ResolvedSource:
    source: SourceSpec
    table_name: str
    requested_time_size: int
    requested_z_size: int
    requested_y_size: int
    requested_x_size: int


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


def _ordering_fingerprint(table: pa.Table) -> str:
    if table.num_rows == 0:
        return "empty"

    preferred_keys = ["prepared_id", "tile_name", "time_start", "z_start", "y_start", "x_start"]
    present = [name for name in preferred_keys if name in table.column_names]
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


def build_table_stats(table: pa.Table, resolved: ResolvedSource) -> TableStats:
    return TableStats(
        num_rows=int(table.num_rows),
        max_time_size=_safe_max(table, "time_size", resolved.source.time_size or 0),
        max_z_size=_safe_max(table, "z_size", resolved.source.z_size or 0),
        max_y_size=_safe_max(table, "y_size", resolved.source.y_size or 0),
        max_x_size=_safe_max(table, "x_size", resolved.source.x_size or 0),
        max_channel_size=_safe_max(table, "channel_size", 0),
        ordering_fingerprint=_ordering_fingerprint(table),
        sample_type=resolved.source.sample_type.value,
        fixed_axes=resolved.source.fixed_axes,
        dynamic_axes=resolved.source.dynamic_axes,
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

    @classmethod
    def _mapping_value_text_expr(cls, value_expr: str) -> str:
        return f"lower(trim(both '\"' from {value_expr}::text))"

    @classmethod
    def _channel_key_expr(cls, alias: str, channel_pattern: str) -> str:
        pattern_sql = cls._sql_string(_normalize_channel_token(channel_pattern))
        return (
            "("
            f"SELECT mapping.key "
            f"FROM jsonb_each(COALESCE({alias}.channel_mapping, '{{}}'::jsonb)) AS mapping(key, value) "
            f"WHERE {cls._mapping_value_text_expr('mapping.value')} ~ {pattern_sql} "
            f"ORDER BY mapping.key "
            f"LIMIT 1"
            ")"
        )

    @classmethod
    def _channel_key_matches_clause(cls, alias: str, key: str, pattern: str) -> str:
        key_sql = cls._sql_string(str(key))
        pattern_sql = cls._sql_string(_normalize_channel_token(pattern))
        return (
            "EXISTS ("
            f"SELECT 1 "
            f"FROM jsonb_each(COALESCE({alias}.channel_mapping, '{{}}'::jsonb)) AS mapping(key, value) "
            f"WHERE mapping.key = {key_sql} "
            f"  AND {cls._mapping_value_text_expr('mapping.value')} ~ {pattern_sql}"
            ")"
        )

    @classmethod
    def _any_channel_matches_clause(cls, alias: str, pattern: str) -> str:
        pattern_sql = cls._sql_string(_normalize_channel_token(pattern))
        return (
            "EXISTS ("
            f"SELECT 1 "
            f"FROM jsonb_each(COALESCE({alias}.channel_mapping, '{{}}'::jsonb)) AS mapping(key, value) "
            f"WHERE {cls._mapping_value_text_expr('mapping.value')} ~ {pattern_sql}"
            ")"
        )

    @staticmethod
    def _channel_cdf_expr(alias: str, channel: int, target: str) -> str:
        return (
            f"(CASE "
            f"WHEN jsonb_typeof({alias}.channels_metadata -> '{channel}' -> 'cdf_{target}') = 'array' "
            f"THEN (SELECT min(value::integer) "
            f"FROM jsonb_array_elements_text("
            f"COALESCE({alias}.channels_metadata -> '{channel}' -> 'cdf_{target}', '[]'::jsonb)"
            f") AS cdf(value)) "
            f"ELSE ({alias}.channels_metadata -> '{channel}' ->> 'cdf_{target}')::integer "
            f"END)"
        )

    @staticmethod
    def _channel_cdf_expr_for_key(alias: str, channel_key_sql: str, target: str) -> str:
        return (
            f"(CASE "
            f"WHEN jsonb_typeof({alias}.channels_metadata -> ({channel_key_sql}) -> 'cdf_{target}') = 'array' "
            f"THEN (SELECT min(value::integer) "
            f"FROM jsonb_array_elements_text("
            f"COALESCE({alias}.channels_metadata -> ({channel_key_sql}) -> 'cdf_{target}', '[]'::jsonb)"
            f") AS cdf(value)) "
            f"ELSE ({alias}.channels_metadata -> ({channel_key_sql}) ->> 'cdf_{target}')::integer "
            f"END)"
        )

    @classmethod
    def build_non_channel_where_sql(cls, source: SourceSpec, query: QuerySpec, alias: str = "s") -> str:
        query.validate()
        clauses = ["1=1"]

        if query.roi_list:
            clauses.append(f"{alias}.prepared_id IN {cls._sql_list(query.roi_list)}")

        if query.tile_list:
            clauses.append(f"{alias}.tile_name IN {cls._sql_list(query.tile_list)}")

        if query.timepoint_list:
            clauses.append(f"{alias}.time_start IN {cls._sql_list(query.timepoint_list)}")

        if query.holdout_split == "train":
            clauses.append(f"COALESCE({alias}.is_test_split, false) = false")
        elif query.holdout_split == "test":
            clauses.append(f"COALESCE({alias}.is_test_split, false) = true")

        for location_key in query.required_locations or ():
            clauses.append(f"COALESCE({alias}.{location_key}, false) = true")

        if query.synthetic_only:
            clauses.append(f"COALESCE({alias}.is_synthetic, false) = true")

        if source.has_annotation_payload:
            clauses.append(f"{alias}.annotation_count > 0")

        if query.channel_count is not None:
            clauses.append(f"COALESCE({alias}.channel_size, 0) = {int(query.channel_count)}")
        if query.min_channel_count is not None:
            clauses.append(f"COALESCE({alias}.channel_size, 0) >= {int(query.min_channel_count)}")
        if query.max_channel_count is not None:
            clauses.append(f"COALESCE({alias}.channel_size, 0) <= {int(query.max_channel_count)}")

        return " AND ".join(clauses)

    @classmethod
    def build_channel_where_clauses(cls, source: SourceSpec, query: QuerySpec, alias: str = "s") -> list[str]:
        clauses: list[str] = []

        for key, pattern in (query.required_channel_key_patterns or {}).items():
            clauses.append(cls._channel_key_matches_clause(alias, key, pattern))

        for localization in query.required_channel_localizations or ():
            clauses.append(cls._any_channel_matches_clause(alias, _literal_channel_pattern(localization)))

        if query.any_channel_patterns:
            any_clauses = [cls._any_channel_matches_clause(alias, pattern) for pattern in query.any_channel_patterns]
            clauses.append("(" + " OR ".join(any_clauses) + ")")

        if query.all_channel_patterns:
            for pattern in query.all_channel_patterns:
                clauses.append(cls._any_channel_matches_clause(alias, pattern))

        if query.cdf_threshold is not None:
            if not source.supports_metric_filters:
                raise NotImplementedError("cdf_threshold is not supported for tile sources")
            threshold = float(query.cdf_threshold)
            if query.cdf_threshold_channel_localizations:
                for localization in query.cdf_threshold_channel_localizations:
                    channel_key_sql = cls._channel_key_expr(alias, localization)
                    clauses.append(f"{channel_key_sql} IS NOT NULL")
                    clauses.append(
                        f"{cls._channel_cdf_expr_for_key(alias, channel_key_sql, query.cdf_target)} >= {threshold}"
                    )
            else:
                raise NotImplementedError(
                    "cdf_threshold_channel_localizations is required for cube metric filters"
                )

        return clauses

    @classmethod
    def build_missing_channel_mapping_where_sql(
        cls,
        source: SourceSpec,
        query: QuerySpec,
        alias: str = "s",
    ) -> Optional[str]:
        has_channel_requirements = bool(
            query.cdf_threshold_channel_localizations
            or query.required_channel_localizations
            or query.any_channel_patterns
            or query.all_channel_patterns
            or query.required_channel_key_patterns
        )
        if not has_channel_requirements:
            return None

        return f"{alias}.channel_mapping IS NULL"

    @classmethod
    def build_where_sql(cls, source: SourceSpec, query: QuerySpec, alias: str = "s") -> str:
        clauses = [cls.build_non_channel_where_sql(source, query, alias=alias)]
        clauses.extend(cls.build_channel_where_clauses(source, query, alias=alias))
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
    def _fetch_valid_channel_localizations(cls, db_client) -> set[str]:
        table = db_client.execute_arrow(
            """
            SELECT localization
            FROM fish_db.tags
            WHERE localization IS NOT NULL
              AND COALESCE(is_deleted, false) = false
            """
        )
        valid: set[str] = set()
        if "localization" not in table.column_names:
            return valid
        for raw in table["localization"].to_pylist():
            valid.update(cls._iter_localization_tokens(raw))
        return valid

    @classmethod
    def _loader_channel_config(
        cls,
        config,
        db_client=None,
    ) -> Optional[tuple[str, ...]]:
        raw_selected = getattr(config.datasets, "selected_channel_localizations", None)
        selected_localizations = cls._normalized_channel_localizations(raw_selected)

        # NOTE: uncomment when new DB image has updated tables
        # if db_client is not None and selected_localizations:
        #     valid_localizations = cls._fetch_valid_channel_localizations(db_client)
        #     invalid = sorted(set(selected_localizations) - valid_localizations)
        #     if invalid:
        #         raise ValueError(
        #             "Unknown selected channel localizations "
        #             f"{invalid}; expected values drawn from fish_db.tags.localization"
        #         )

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

        # NOTE: uncomment when new DB image has updated tables
        # if db_client is not None and required:
        #     valid_localizations = cls._fetch_valid_channel_localizations(db_client)
        #     invalid = sorted(set(required) - valid_localizations)
        #     if invalid:
        #         raise ValueError(
        #             "Unknown required channel localizations "
        #             f"{invalid}; expected values drawn from fish_db.tags.localization"
        #         )

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
        return QuerySpec(
            roi_list=cls._to_tuple(config.datasets.roi_list),
            tile_list=cls._to_tuple(config.datasets.tile_list),
            timepoint_list=cls._to_tuple(config.datasets.timepoint_list),
            max_rows=config.datasets.max_hypercubes,
            cdf_threshold=config.datasets.cdf_threshold,
            cdf_target=config.datasets.cdf_target,
            cdf_threshold_channel_localizations=cls._to_tuple(getattr(
                config.datasets.databases,
                "cdf_threshold_channel_localizations",
                None,
            )),
            required_locations=cls._to_tuple(getattr(config.datasets.databases, "required_locations", None)),
            holdout_split=getattr(config.datasets.databases, "holdout_split", None),
            synthetic_only=bool(config.datasets.synthetic_only),
            channel_count=getattr(config.datasets.databases, "channel_count", None),
            min_channel_count=getattr(config.datasets.databases, "min_channel_count", None),
            max_channel_count=getattr(config.datasets.databases, "max_channel_count", None),
            required_channel_key_patterns=cls._to_dict(getattr(
                config.datasets.databases, "required_channel_key_patterns", None
            )),
            any_channel_patterns=cls._to_tuple(getattr(config.datasets.databases, "any_channel_patterns", None)),
            all_channel_patterns=cls._to_tuple(getattr(config.datasets.databases, "all_channel_patterns", None)),
            required_channel_localizations=cls._required_channel_localizations_from_config(
                config,
                db_client=db_client,
            ),
        )

    @staticmethod
    def build_store_spec_from_config(config) -> StoreSpec:
        return StoreSpec(root_dir=str(config.datasets.databases.node_local_store_root))

    @classmethod
    def _concrete_source(
        cls,
        base: SourceSpec,
        source_key: str,
        time_size: Optional[int],
        z_size: Optional[int],
        y_size: Optional[int],
        x_size: Optional[int],
    ) -> SourceSpec:
        return SourceSpec(
            key=source_key,
            training_table_kind=base.training_table_kind,
            table_name_template=base.table_name_template,
            fixed_axes=base.fixed_axes,
            dynamic_axes=base.dynamic_axes,
            time_size=time_size,
            z_size=z_size,
            y_size=y_size,
            x_size=x_size,
            order_by=base.order_by,
        )

    @staticmethod
    def _materialize_table_name(source: SourceSpec) -> str:
        return source.table_name_template.format(
            time_size=source.time_size,
            z_size=source.z_size,
            y_size=source.y_size,
            x_size=source.x_size,
        )

    @classmethod
    def resolve_from_config(cls, config, *, db_client=None) -> ResolvedSource:
        requested_time, requested_z, requested_y, requested_x = cls._requested_shape(config)
        has_annotations = bool(config.datasets.has_annotations)
        sample_type = cls._sample_type(config)

        if sample_type == SampleType.CUBE:
            base_key = "cube_with_annotations" if has_annotations else "cube_without_annotations"
            t, z, y, x = requested_time, requested_z, requested_y, requested_x
        else:
            base_key = "tile_with_annotations" if has_annotations else "tile_without_annotations"
            t, z, y, x = requested_time, None, None, None

        base = SOURCE_TABLES[base_key]
        source_key = base.table_name_template.format(
            time_size=t, z_size=z, y_size=y, x_size=x,
        )
        source = cls._concrete_source(
            base, source_key=source_key,
            time_size=t, z_size=z, y_size=y, x_size=x,
        )

        table_name = cls._materialize_table_name(source)
        if db_client is not None:
            db_client.assert_relation_exists(table_name)

        return ResolvedSource(
            source=source,
            table_name=table_name,
            requested_time_size=requested_time,
            requested_z_size=requested_z,
            requested_y_size=requested_y,
            requested_x_size=requested_x,
        )


class SqlQueryPlanner:
    @staticmethod
    def _cube_without_annotations_select(alias: str) -> list[str]:
        return [
            f"{alias}.first_pc_id",
            f"{alias}.prepared_id",
            f"{alias}.tile_name",
            f"{alias}.server_folder",
            f"{alias}.output_folder",
            f"{alias}.is_synthetic",
            f"{alias}.is_available",
            f"{alias}.exists_prfs",
            f"{alias}.exists_aws",
            f"{alias}.exists_oak",
            f"{alias}.exists_abc",
            f"{alias}.time_start",
            f"{alias}.time_size",
            f"{alias}.z_start",
            f"{alias}.y_start",
            f"{alias}.x_start",
            f"{alias}.z_size",
            f"{alias}.y_size",
            f"{alias}.x_size",
            f"{alias}.channel_size",
            f"{alias}.is_complete",
            f"{alias}.is_test_split",
            f"{alias}.channel_mapping",
            f"{alias}.channels_metadata",
            "0::integer AS annotation_count",
            "false AS has_annotations",
            "NULL::jsonb AS annotations_metadata",
        ]

    @staticmethod
    def _cube_with_annotations_select(alias: str) -> list[str]:
        return [
            f"{alias}.first_pc_id",
            f"{alias}.prepared_id",
            f"{alias}.tile_name",
            f"{alias}.server_folder",
            f"{alias}.output_folder",
            f"{alias}.is_synthetic",
            f"{alias}.is_available",
            f"{alias}.exists_prfs",
            f"{alias}.exists_aws",
            f"{alias}.exists_oak",
            f"{alias}.exists_abc",
            f"{alias}.time_start",
            f"{alias}.time_size",
            f"{alias}.z_start",
            f"{alias}.y_start",
            f"{alias}.x_start",
            f"{alias}.z_size",
            f"{alias}.y_size",
            f"{alias}.x_size",
            f"{alias}.channel_size",
            f"{alias}.is_complete",
            f"{alias}.is_test_split",
            f"{alias}.channel_mapping",
            f"{alias}.channels_metadata",
            f"{alias}.annotation_count",
            "true AS has_annotations",
            f"{alias}.annotations_metadata",
        ]

    @staticmethod
    def _tile_with_annotations_select(alias: str) -> list[str]:
        return [
            f"{alias}.first_pc_id",
            f"{alias}.prepared_id",
            f"{alias}.tile_name",
            f"{alias}.server_folder",
            f"{alias}.output_folder",
            f"{alias}.is_synthetic",
            f"{alias}.is_available",
            f"{alias}.exists_prfs",
            f"{alias}.exists_aws",
            f"{alias}.exists_oak",
            f"{alias}.exists_abc",
            f"{alias}.time_start",
            f"{alias}.time_size",
            f"{alias}.z_start",
            f"{alias}.y_start",
            f"{alias}.x_start",
            f"{alias}.z_size",
            f"{alias}.y_size",
            f"{alias}.x_size",
            f"{alias}.channel_size",
            f"{alias}.is_complete",
            f"{alias}.is_test_split",
            f"{alias}.channel_mapping",
            "NULL::jsonb AS channels_metadata",
            f"{alias}.annotation_count",
            "true AS has_annotations",
            f"{alias}.annotations_metadata",
        ]

    @staticmethod
    def _tile_without_annotations_select(alias: str) -> list[str]:
        return [
            f"{alias}.first_pc_id",
            f"{alias}.prepared_id",
            f"{alias}.tile_name",
            f"{alias}.server_folder",
            f"{alias}.output_folder",
            f"{alias}.is_synthetic",
            f"{alias}.is_available",
            f"{alias}.exists_prfs",
            f"{alias}.exists_aws",
            f"{alias}.exists_oak",
            f"{alias}.exists_abc",
            f"{alias}.time_start",
            f"{alias}.time_size",
            f"{alias}.z_start",
            f"{alias}.y_start",
            f"{alias}.x_start",
            f"{alias}.z_size",
            f"{alias}.y_size",
            f"{alias}.x_size",
            f"{alias}.channel_size",
            f"{alias}.is_complete",
            f"{alias}.is_test_split",
            f"{alias}.channel_mapping",
            "NULL::jsonb AS channels_metadata",
            "0::integer AS annotation_count",
            "false AS has_annotations",
            "NULL::jsonb AS annotations_metadata",
        ]

    @classmethod
    def build_sql(cls, resolved: ResolvedSource, query: QuerySpec) -> str:
        alias = "s"
        source = resolved.source

        if source.training_table_kind == TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS:
            select_items = cls._cube_without_annotations_select(alias)
        elif source.training_table_kind == TrainingTableKind.CUBE_WITH_ANNOTATIONS:
            select_items = cls._cube_with_annotations_select(alias)
        elif source.training_table_kind == TrainingTableKind.TILE_WITH_ANNOTATIONS:
            select_items = cls._tile_with_annotations_select(alias)
        elif source.training_table_kind == TrainingTableKind.TILE_WITHOUT_ANNOTATIONS:
            select_items = cls._tile_without_annotations_select(alias)
        else:
            raise NotImplementedError(source.training_table_kind)

        select_sql = ",\n                ".join(select_items)
        where_sql = FilterBuilder.build_where_sql(source, query, alias=alias)
        order_sql = ", ".join(f"{alias}.{name}" for name in source.order_by)

        sql = f"""
            SELECT
                row_number() OVER (ORDER BY {order_sql}) - 1 AS row_id,
                {select_sql}
            FROM public.{resolved.table_name} {alias}
            WHERE {where_sql}
            ORDER BY {order_sql}
        """
        if query.max_rows is not None:
            sql += f"\nLIMIT {int(query.max_rows)}"
        return sql

    @classmethod
    def build_missing_channel_mapping_diagnostic_sql(
        cls,
        resolved: ResolvedSource,
        query: QuerySpec,
    ) -> Optional[str]:
        alias = "s"
        where_sql = FilterBuilder.build_missing_channel_mapping_where_sql(
            resolved.source,
            query,
            alias=alias,
        )
        if where_sql is None:
            return None

        return f"""
            SELECT
                count(*) AS dropped_row_count,
                count(DISTINCT {alias}.prepared_id) AS dropped_prepared_count
            FROM public.{resolved.table_name} {alias}
            WHERE {where_sql}
        """


class MappedTable:
    def __init__(self, descriptor: MappedTableDescriptor) -> None:
        self.descriptor = descriptor
        self._source = None
        self._reader = None
        self._table: Optional[pa.Table] = None

    @staticmethod
    def _remap_server_folder(table: pa.Table, query: QuerySpec, server_path_override: Optional[str] = None) -> pa.Table:
        """Replace server_folder with the path for the target storage location."""
        if "server_folder" not in table.column_names:
            raise ValueError("server_folder column not found in table")
        locations = query.required_locations
        if not locations:
            raise ValueError(f"server_folder remapping requires exactly one required_location, got {locations!r}")
        if len(locations) != 1:
            raise ValueError(
                f"server_folder remapping requires exactly one required_location, got {locations!r}"
            )
        location_key = locations[0]
        server_path = LOCATION_SERVER_PATHS.get(location_key)
        if server_path_override is not None:
            server_path = server_path_override
        if server_path is None:
            raise ValueError(
                f"No server path configured for location {location_key!r}. "
                f"Known locations: {sorted(LOCATION_SERVER_PATHS)}"
            )
        col_idx = table.column_names.index("server_folder")
        new_col = pa.array([server_path] * table.num_rows, type=pa.utf8())
        table = table.set_column(col_idx, "server_folder", new_col)
        logger.info(
            "[MappedTable] remapped server_folder to %s for location=%s (%d rows)",
            server_path,
            location_key,
            table.num_rows,
        )
        return table

    @staticmethod
    def _root(node_id: str, resolved: ResolvedSource, query: QuerySpec, store: StoreSpec) -> Path:
        payload = {
            "node_id": node_id,
            "source_key": resolved.source.key,
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
    ) -> "MappedTable":
        total_t0 = time.perf_counter()
        root = cls._root(node_id=node_id, resolved=resolved, query=query, store=store)
        root.mkdir(parents=True, exist_ok=True)

        descriptor_path = root / "descriptor.json"
        sample_path = root / "sample_table.arrow"

        if local_rank == 0:
            diagnostic_sql = SqlQueryPlanner.build_missing_channel_mapping_diagnostic_sql(resolved, query)
            if diagnostic_sql is not None:
                if diagnostic_verbose:
                    logger.warning(
                        "[MappedTable] running channel-mapping diagnostic query for %s; "
                        "this may be slow on large tables (100s+). "
                        "Set datasets.databases.diagnostic_verbose=false to skip.",
                        resolved.table_name,
                    )
                    diagnostic_t0 = time.perf_counter()
                    diagnostic_table = db_client.execute_arrow(diagnostic_sql)
                    logger.info(
                        "[MappedTable] diagnostic query for %s completed in %.2fs",
                        resolved.table_name,
                        time.perf_counter() - diagnostic_t0,
                    )
                    dropped_rows = int(diagnostic_table["dropped_row_count"][0].as_py() or 0)
                    dropped_prepared = int(diagnostic_table["dropped_prepared_count"][0].as_py() or 0)
                    if dropped_rows > 0:
                        logger.warning(
                            "[MappedTable] filtering out %s rows across %s prepared_ids from %s because "
                            "channel_mapping is missing.",
                            dropped_rows,
                            dropped_prepared,
                            resolved.table_name,
                        )
                else:
                    logger.info(
                        "[MappedTable] skipping channel-mapping diagnostic query for %s "
                        "(set datasets.databases.diagnostic_verbose=true to enable)",
                        resolved.table_name,
                    )
            sql = SqlQueryPlanner.build_sql(resolved, query)
            fetch_t0 = time.perf_counter()
            table = db_client.execute_arrow(sql)
            table = cls._remap_server_folder(table, query, server_path_override=server_path_override)
            fetch_elapsed = time.perf_counter() - fetch_t0
            logger.info(
                "[MappedTable] fetched %s rows from %s in %.2fs",
                table.num_rows,
                resolved.table_name,
                fetch_elapsed,
            )
            write_t0 = time.perf_counter()
            with pa.OSFile(str(sample_path), "wb") as sink:
                with pa_ipc.new_file(sink, table.schema) as writer:
                    writer.write(table)
            logger.info(
                "[MappedTable] wrote sample table for %s to %s in %.2fs",
                resolved.table_name,
                sample_path,
                time.perf_counter() - write_t0,
            )

            descriptor = MappedTableDescriptor(
                sample_table=SharedTableDescriptor(
                    path=str(sample_path),
                    num_rows=int(table.num_rows),
                    source_key=resolved.source.key,
                ),
                stats=build_table_stats(table, resolved),
            )
            descriptor_path.write_text(json.dumps(asdict(descriptor)))
            logger.info(
                "[MappedTable] descriptor for %s materialized in %.2fs",
                resolved.table_name,
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
            resolved.table_name,
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
