"""Unit tests for local_metadata_store helpers (no live Postgres)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from cell_observatory_platform.data.databases.local_metadata_store import (
    LOCATION_SERVER_PATHS,
    SOURCE_TABLES,
    FilterBuilder,
    MappedTable,
    QuerySpec,
    ResolvedSource,
    SqlQueryPlanner,
    TableResolver,
    TrainingTableKind,
)


def test_remap_server_folder_override_allows_unknown_location():
    table = pa.table({"server_folder": pa.array(["/legacy/a", "/legacy/b"])})
    query = QuerySpec(required_locations=["exists_oak"])

    with pytest.raises(ValueError, match="No server path configured"):
        MappedTable._remap_server_folder(table, query)

    override = "/custom/server/root"
    out = MappedTable._remap_server_folder(table, query, server_path_override=override)
    assert out["server_folder"].to_pylist() == [override, override]


def test_remap_server_folder_override_replaces_catalog_path():
    table = pa.table({"server_folder": pa.array(["/ignored"])})
    query = QuerySpec(required_locations=["exists_abc"])
    catalog = LOCATION_SERVER_PATHS["exists_abc"]

    mapped = MappedTable._remap_server_folder(table, query)
    assert mapped["server_folder"][0].as_py() == catalog

    override = "/override/replaces/catalog"
    remapped = MappedTable._remap_server_folder(table, query, server_path_override=override)
    assert remapped["server_folder"][0].as_py() == override


# ---------------------------------------------------------------------------
# Query-planner SQL assembly (DB-free; assert on stable SQL substrings).
#
# These tests intentionally check small, drift-catching fragments rather than
# the whole SQL string so cosmetic reformatting does not break them while a
# real semantic change (a dropped/renamed column, a changed filter predicate)
# does.
# ---------------------------------------------------------------------------


# Map each TrainingTableKind to a registry entry whose concrete source we can
# materialize without a DB. The cube kinds support metric (cdf) filters; the
# tile kinds do not.
_KIND_TO_REGISTRY_KEY = {
    TrainingTableKind.CUBE_WITH_ANNOTATIONS: "cube_with_annotations",
    TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS: "cube_without_annotations",
    TrainingTableKind.TILE_WITH_ANNOTATIONS: "tile_with_annotations",
    TrainingTableKind.TILE_WITHOUT_ANNOTATIONS: "tile_without_annotations",
}


def _resolved(kind: TrainingTableKind, table_name: str = "mytbl") -> ResolvedSource:
    """Build a ResolvedSource for ``kind`` without touching a live DB."""
    base = SOURCE_TABLES[_KIND_TO_REGISTRY_KEY[kind]]
    source = TableResolver._concrete_source(
        base,
        source_key=table_name,
        time_size=1,
        z_size=64,
        y_size=64,
        x_size=64,
    )
    return ResolvedSource(
        source=source,
        table_name=table_name,
        requested_time_size=1,
        requested_z_size=64,
        requested_y_size=64,
        requested_x_size=64,
    )


def test_build_where_sql_no_filters_is_true():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    where = FilterBuilder.build_where_sql(resolved.source, QuerySpec())
    assert where == "1=1"


def test_build_where_sql_non_channel_filters_emit_expected_fragments():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    query = QuerySpec(
        roi_list=[1, 2, 3],
        tile_list=["tileA", "tileB"],
        timepoint_list=[5, 7],
        holdout_split="train",
        required_locations=["exists_abc"],
        synthetic_only=True,
        channel_count=2,
        min_channel_count=1,
        max_channel_count=4,
    )
    where = FilterBuilder.build_where_sql(resolved.source, query)

    assert "s.prepared_id IN (1,2,3)" in where
    assert "s.tile_name IN ('tileA','tileB')" in where
    assert "s.time_start IN (5,7)" in where
    assert "COALESCE(s.is_test_split, false) = false" in where
    assert "COALESCE(s.exists_abc, false) = true" in where
    assert "COALESCE(s.is_synthetic, false) = true" in where
    assert "COALESCE(s.channel_size, 0) = 2" in where
    assert "COALESCE(s.channel_size, 0) >= 1" in where
    assert "COALESCE(s.channel_size, 0) <= 4" in where


def test_build_where_sql_holdout_test_branch():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    where = FilterBuilder.build_where_sql(
        resolved.source, QuerySpec(holdout_split="test")
    )
    assert "COALESCE(s.is_test_split, false) = true" in where


def test_build_where_sql_with_annotations_source_requires_annotation_count():
    resolved = _resolved(TrainingTableKind.CUBE_WITH_ANNOTATIONS)
    where = FilterBuilder.build_where_sql(resolved.source, QuerySpec())
    assert "s.annotation_count > 0" in where


def test_build_where_sql_respects_alias():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    where = FilterBuilder.build_where_sql(
        resolved.source, QuerySpec(roi_list=[1]), alias="t"
    )
    assert "t.prepared_id IN (1)" in where
    assert "s.prepared_id" not in where


def test_build_where_sql_channel_localization_and_patterns():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    query = QuerySpec(
        required_channel_localizations=["Nucleus"],
        required_channel_key_patterns={"ch0": "dapi"},
        any_channel_patterns=["foo", "bar"],
    )
    clauses = FilterBuilder.build_channel_where_clauses(resolved.source, query)
    joined = " ".join(clauses)

    # localization patterns are normalized to lowercase and matched against the
    # jsonb channel_mapping values.
    assert "jsonb_each(COALESCE(s.channel_mapping, '{}'::jsonb))" in joined
    assert "~ 'nucleus'" in joined
    # explicit key pattern restricts mapping.key.
    assert "mapping.key = 'ch0'" in joined
    # any_channel_patterns are OR'd together inside a single clause.
    assert any(" OR " in clause for clause in clauses)


def test_cdf_threshold_requires_cube_source():
    resolved = _resolved(TrainingTableKind.TILE_WITHOUT_ANNOTATIONS)
    query = QuerySpec(
        cdf_threshold=0.5,
        cdf_threshold_channel_localizations=["membrane"],
    )
    with pytest.raises(NotImplementedError):
        FilterBuilder.build_channel_where_clauses(resolved.source, query)


def test_cdf_threshold_without_localizations_raises():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    with pytest.raises(NotImplementedError):
        FilterBuilder.build_channel_where_clauses(
            resolved.source, QuerySpec(cdf_threshold=0.5)
        )


def test_cdf_threshold_emits_resolution_and_threshold_clauses():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    query = QuerySpec(
        cdf_threshold=0.5,
        cdf_target="95",
        cdf_threshold_channel_localizations=["membrane"],
    )
    clauses = FilterBuilder.build_channel_where_clauses(resolved.source, query)
    joined = " ".join(clauses)
    # one clause asserts the channel key resolves, the next applies the cdf cut.
    assert "IS NOT NULL" in joined
    assert "cdf_95" in joined
    assert ">= 0.5" in joined


@pytest.mark.parametrize(
    "kind",
    [
        TrainingTableKind.CUBE_WITH_ANNOTATIONS,
        TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS,
        TrainingTableKind.TILE_WITH_ANNOTATIONS,
        TrainingTableKind.TILE_WITHOUT_ANNOTATIONS,
    ],
)
def test_build_sql_common_skeleton(kind):
    resolved = _resolved(kind, table_name="prepared_demo")
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec())

    assert "row_number() OVER (ORDER BY" in sql
    assert ") - 1 AS row_id" in sql
    assert "FROM public.prepared_demo s" in sql
    assert "\n            WHERE " in sql
    assert "ORDER BY s.prepared_id" in sql
    # every kind selects these core columns.
    assert "s.first_pc_id" in sql
    assert "s.server_folder" in sql
    assert "s.channel_mapping" in sql


def test_build_sql_cube_with_annotations_columns():
    resolved = _resolved(TrainingTableKind.CUBE_WITH_ANNOTATIONS)
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec())
    assert "s.channels_metadata" in sql
    assert "s.annotation_count" in sql
    assert "true AS has_annotations" in sql
    assert "s.annotations_metadata" in sql
    # the with-annotations variant must NOT emit the stub literals.
    assert "0::integer AS annotation_count" not in sql
    assert "false AS has_annotations" not in sql


def test_build_sql_cube_without_annotations_emits_stub_columns():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec())
    assert "s.channels_metadata" in sql
    assert "0::integer AS annotation_count" in sql
    assert "false AS has_annotations" in sql
    assert "NULL::jsonb AS annotations_metadata" in sql


def test_build_sql_tile_with_annotations_columns():
    resolved = _resolved(TrainingTableKind.TILE_WITH_ANNOTATIONS)
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec())
    # tile sources have no real channels_metadata; it is stubbed to NULL.
    assert "NULL::jsonb AS channels_metadata" in sql
    assert "s.annotation_count" in sql
    assert "true AS has_annotations" in sql
    assert "0::integer AS annotation_count" not in sql


def test_build_sql_tile_without_annotations_emits_stub_columns():
    resolved = _resolved(TrainingTableKind.TILE_WITHOUT_ANNOTATIONS)
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec())
    assert "NULL::jsonb AS channels_metadata" in sql
    assert "0::integer AS annotation_count" in sql
    assert "false AS has_annotations" in sql
    assert "NULL::jsonb AS annotations_metadata" in sql


def test_build_sql_appends_limit_when_max_rows_set():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    assert "LIMIT" not in SqlQueryPlanner.build_sql(resolved, QuerySpec())
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec(max_rows=25))
    assert "LIMIT 25" in sql


def test_build_sql_embeds_where_filters():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    sql = SqlQueryPlanner.build_sql(
        resolved, QuerySpec(roi_list=[42], timepoint_list=[3])
    )
    assert "WHERE s.prepared_id IN (42) AND s.time_start IN (3)" in sql


def test_missing_channel_mapping_diagnostic_none_without_channel_requirements():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS)
    assert (
        SqlQueryPlanner.build_missing_channel_mapping_diagnostic_sql(
            resolved, QuerySpec()
        )
        is None
    )


def test_missing_channel_mapping_diagnostic_present_with_requirements():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS, table_name="demo")
    sql = SqlQueryPlanner.build_missing_channel_mapping_diagnostic_sql(
        resolved, QuerySpec(required_channel_localizations=["nucleus"])
    )
    assert sql is not None
    assert "count(*) AS dropped_row_count" in sql
    assert "count(DISTINCT s.prepared_id) AS dropped_prepared_count" in sql
    assert "FROM public.demo s" in sql
    assert "WHERE s.channel_mapping IS NULL" in sql


def test_filter_diagnostic_sql_builds_cumulative_union():
    resolved = _resolved(TrainingTableKind.CUBE_WITHOUT_ANNOTATIONS, table_name="demo")
    query = QuerySpec(roi_list=[1], timepoint_list=[2], holdout_split="train")
    sql, labels = SqlQueryPlanner.build_filter_diagnostic_sql(resolved, query)

    # baseline + one step per labeled clause.
    assert labels[0] == "baseline (no filter)"
    assert "roi_list=[1]" in labels
    assert "timepoint_list=[2]" in labels
    assert "holdout_split=train" in labels
    assert len(labels) == 4

    assert "SELECT 0 AS step, count(*)::bigint AS n_rows FROM public.demo s" in sql
    assert "UNION ALL" in sql
    assert sql.rstrip().endswith("ORDER BY step")
    # the cumulative WHERE for the last step contains all earlier clauses.
    assert "s.prepared_id IN (1) AND s.time_start IN (2)" in sql
