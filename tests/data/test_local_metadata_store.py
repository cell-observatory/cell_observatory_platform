"""Unit tests for local_metadata_store helpers (no live Postgres).

Covers the api.cube_training / api.tiles_training views: view resolution, the
single projection, the api.roi_channels semi-join and CDF predicates, and the
config -> QuerySpec / StoreSpec mapping.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cell_observatory_platform.data.databases.local_metadata_store import (
    CHANNEL_CATALOG,
    TRAINING_VIEWS,
    TableResolver,
    _assert_requested_time_size,
    _selected_channel_size,
    FilterBuilder,
    MappedTable,
    QuerySpec,
    ResolvedSource,
    SampleIndexPlanner,
    SampleType,
    SqlQueryPlanner,
    channel_role,
    is_object_role,
)
from cell_observatory_platform.data.databases.schema import (
    required_columns,
    validate_non_null,
    validate_projection,
)


def _resolved(
    sample_type: SampleType = SampleType.CUBE,
    *,
    with_targets: bool = False,
    z: int = 64,
) -> ResolvedSource:
    return ResolvedSource(
        view=TRAINING_VIEWS[sample_type],
        requested_time_size=1,
        requested_z_size=z,
        requested_y_size=64,
        requested_x_size=64,
        with_targets=with_targets,
        location_id=3,
        location_name="synthetic",
    )


# --------------------------------------------------------------------------- #
# view registry / identity
# --------------------------------------------------------------------------- #


def test_tile_view_name_is_plural():
    """api.tiles_training, not api.tile_training."""
    assert TRAINING_VIEWS[SampleType.TILE].qualified_name == "api.tiles_training"
    assert TRAINING_VIEWS[SampleType.CUBE].qualified_name == "api.cube_training"


def test_cube_order_by_is_the_db_natural_key():
    """Ordering is the UNIQUE-constrained natural key, NOT serial cube_id.

    row_id comes from row_number() OVER (ORDER BY ...), and skip_batches resume
    replays a row RANGE -- so a non-unique or unstable order silently replays
    different data. cube_id is a serial and is not stable across a repopulate.
    """
    order_by = TRAINING_VIEWS[SampleType.CUBE].order_by
    assert order_by == (
        "source_kind", "roi_id", "tile_id",
        "time_start", "z_start", "y_start", "x_start",
    )
    assert "cube_id" not in order_by


def test_axis_classification_per_sample_type():
    """fixed/dynamic axes drive BOTH the SQL shape predicate and _shape_from_stats'
    buffer sizing, so the two can never disagree. This is what keeps the
    two-shape / Resize contract intact."""
    cube = TRAINING_VIEWS[SampleType.CUBE]
    tile = TRAINING_VIEWS[SampleType.TILE]
    assert cube.fixed_axes == ("T", "Z", "Y", "X") and cube.dynamic_axes == ("C",)
    assert tile.fixed_axes == () and tile.dynamic_axes == ("T", "Z", "Y", "X", "C")
    assert cube.has_occupancy_metrics and not tile.has_occupancy_metrics


def test_cache_key_includes_shape_and_location():
    """Shape is a predicate, not part of the relation name, so it has to be
    hashed into the cache key explicitly: two runs at different cube sizes must
    not collide on one Arrow file."""
    assert _resolved(z=64).cache_key != _resolved(z=128).cache_key


# --------------------------------------------------------------------------- #
# schema contract
# --------------------------------------------------------------------------- #


def test_required_columns_targets_toggle():
    with_t = required_columns(with_targets=True)
    without_t = required_columns(with_targets=False)
    assert "annotations_metadata" in with_t
    assert "annotations_metadata" not in without_t


def test_required_columns_cube_only_passthrough():
    assert "missing_timepoints" in required_columns(with_targets=False, cube=True)
    assert "missing_timepoints" not in required_columns(with_targets=False, cube=False)


def test_validate_projection_names_missing_columns():
    table = pa.table({"row_id": pa.array([0])})
    with pytest.raises(ValueError, match="tile_relative_path"):
        validate_projection(table, with_targets=False, where="unit")


def test_validate_projection_ignores_extra_columns():
    """The view may grow; do not "fix" a failure by making this a set equality."""
    cols = {c: pa.array([0]) for c in required_columns(with_targets=False)}
    cols["some_new_db_column"] = pa.array([1])
    validate_projection(pa.table(cols), with_targets=False, where="unit")


def test_validate_non_null_rejects_missing_extent():
    """tiles_training derives z/y/x_size from nullable array_shape elements; a
    None there reaches _shape_from_stats and _slice_hypercube."""
    table = pa.table({"z_size": pa.array([64, None], type=pa.int64())})
    with pytest.raises(ValueError, match="z_size"):
        validate_non_null(table, where="unit")


# --------------------------------------------------------------------------- #
# projection + shape-as-predicate
# --------------------------------------------------------------------------- #


def test_build_sql_projects_schema_contract():
    """One projection for every sample type, with no stub columns: each name in
    the schema contract is selected from the view itself.
    """
    sql = SqlQueryPlanner.build_sql(_resolved(), QuerySpec())
    assert "FROM api.cube_training s" in sql
    for column in required_columns(with_targets=False):
        if column == "row_id":
            continue
        assert column in sql
    assert "0::integer AS annotation_count" not in sql
    assert "NULL::jsonb" not in sql


def test_build_sql_omits_target_columns_without_annotations():
    """annotations_metadata is the bulk of the payload and the pretrain path
    never reads it."""
    assert "annotations_metadata" not in SqlQueryPlanner.build_sql(
        _resolved(with_targets=False), QuerySpec()
    )
    assert "annotations_metadata" in SqlQueryPlanner.build_sql(
        _resolved(with_targets=True), QuerySpec()
    )


def test_shape_is_a_predicate_not_a_relation_name():
    """Shape lives in a WHERE clause, not in the relation name -- one view per
    sample type serves every cube size."""
    sql = SqlQueryPlanner.build_sql(_resolved(z=128), QuerySpec())
    assert "s.z_size = 128" in sql
    assert "prepared_cube_channel_agg" not in sql


def test_shape_clauses_follow_fixed_axes():
    """Not `if sample_type is CUBE`: when tile_time_windows gains {16} the tile
    view gets fixed_axes=("T",) and the predicate follows with no code change."""
    cube = FilterBuilder._shape_clauses(_resolved(SampleType.CUBE), "s")
    assert [label for label, _ in cube] == [
        "time_size=1", "z_size=64", "y_size=64", "x_size=64",
    ]
    assert FilterBuilder._shape_clauses(_resolved(SampleType.TILE), "s") == []


def test_shape_appears_in_filter_diagnostic():
    """Recovers the error quality lost with assert_relation_exists: a bad shape
    is a NAMED 0-row step rather than a bare empty result."""
    _sql, labels = SqlQueryPlanner.build_filter_diagnostic_sql(
        _resolved(z=128), QuerySpec()
    )
    assert "z_size=128" in labels


def test_sql_joins_storage_locations_for_root():
    """storage_root comes from dry_lab.storage_locations, joined on the
    configured location id -- never from a hardcoded path table."""
    sql = SqlQueryPlanner.build_sql(_resolved(), QuerySpec())
    assert "JOIN dry_lab.storage_locations loc" in sql
    assert "loc.id = 3" in sql
    assert "loc.root_path AS storage_root" in sql
    assert "= ANY(s.present_locations)" in sql


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #


def test_annotations_predicate_uses_has_annotations():
    """`annotation_count` is a jsonb MAP keyed by timepoint, not a scalar, so
    `annotation_count > 0` is not a valid predicate. has_annotations is the real
    boolean and is btree-indexed."""
    sql = SqlQueryPlanner.build_sql(_resolved(with_targets=True), QuerySpec())
    assert "s.has_annotations" in sql
    assert "annotation_count >" not in sql


def test_roi_ids_filter_targets_roi_id_column():
    where = FilterBuilder.build_where_sql(_resolved(), QuerySpec(roi_ids=[1, 2, 3]))
    assert "s.roi_id IN (1,2,3)" in where
    assert "prepared_id" not in where


def test_channel_count_filters_use_data_channel_count():
    """The old channel_count included the mask channel, so a config meaning
    "2 signal channels" was really asking for 1 signal + 1 mask."""
    where = FilterBuilder.build_where_sql(
        _resolved(), QuerySpec(min_data_channel_count=1, max_data_channel_count=4)
    )
    assert "s.data_channel_count, 0) >= 1" in where
    assert "s.data_channel_count, 0) <= 4" in where
    assert "channel_size" not in where


_SEMIJOIN = "EXISTS (SELECT 1 FROM api.roi_channels rc WHERE rc.roi_id = s.roi_id"


def test_localization_filter_is_a_catalog_semijoin():
    """Channel predicates go through api.roi_channels, whose columns are scalar
    and indexable -- not through the training row's aggregated arrays, which are
    built by array_agg inside the view's LATERAL and cannot be index-scanned."""
    where = FilterBuilder.build_where_sql(
        _resolved(), QuerySpec(required_channel_localizations=["membrane"])
    )
    assert _SEMIJOIN in where
    assert "lower(rc.localization) = 'membrane'" in where


def test_fluorophore_filter_is_case_insensitive():
    """The DB vocabulary is not uniformly cased -- localization is lowercase but
    fluorophore is mixed ("Electra2", "mTFP1", ...). A case-sensitive match on a
    config saying "electra2" would return nothing, silently."""
    where = FilterBuilder.build_where_sql(
        _resolved(), QuerySpec(required_fluorophores=["Electra2"])
    )
    assert "lower(rc.fluorophore) = 'electra2'" in where


def test_required_annotation_type_is_a_catalog_semijoin():
    where = FilterBuilder.build_where_sql(
        _resolved(), QuerySpec(required_annotation_types=["instance"])
    )
    assert "lower(rc.annotation_type) = 'instance'" in where


def test_no_channel_predicate_touches_the_aggregated_arrays():
    """Regression guard for two dead ends, both of which failed SILENTLY:
    channel_tags containment (no `loc:` tokens exist, so it matched nothing) and
    `= ANY(localization)` over the post-aggregation array (full view scan)."""
    where = FilterBuilder.build_where_sql(
        _resolved(),
        QuerySpec(
            required_channel_localizations=["membrane"],
            required_fluorophores=["mstaygold"],
            required_annotation_types=["instance"],
            required_channel_key_patterns={"0": "membrane"},
        ),
    )
    assert "channel_tags" not in where
    assert "@>" not in where
    assert "= ANY(s.localization)" not in where


def test_channel_key_pattern_matches_channel_idx_not_array_position():
    """Pinning a localization to a specific channel must key off rc.channel_idx.

    An earlier version subscripted the aligned array (`s.localization[k + 1]`),
    which is only equivalent when channel_idx is DENSE -- and the schema keeps
    channel_idx as its own column precisely because it need not be. For an ROI
    with a gap that silently pinned the wrong channel.
    """
    where = FilterBuilder.build_where_sql(
        _resolved(), QuerySpec(required_channel_key_patterns={"3": "membrane"})
    )
    assert "rc.channel_idx = 3" in where
    assert "lower(rc.localization) = 'membrane'" in where
    assert "s.localization[" not in where


def test_cdf_clause_is_a_channel_aligned_array_subscript():
    """occupancy_cdf* is "Length C (= channel_size), aligned to channel_idx",
    mask slots null -- so one subscript, no length branch, and no
    mask-channels-sort-last invariant to guard."""
    where = FilterBuilder.build_where_sql(
        _resolved(),
        QuerySpec(cdf_threshold=150, cdf_threshold_channel_localizations=["membrane"]),
    )
    assert "(s.occupancy_cdf90)[array_position(s.localization, 'membrane')] >= 150.0" in where
    assert "jsonb_array_elements_text" not in where
    assert "jsonb_typeof" not in where
    # the "does this ROI have the channel" half is an indexed semi-join; only the
    # per-row slot lookup stays positional, because no join can answer it
    assert "lower(rc.localization) = 'membrane'" in where


def test_cdf_emits_resolution_and_threshold_clauses():
    """Two clauses so the narrowing diagnostic separates "no such channel" from
    "channel too dim" -- a null subscript fails the >= either way."""
    labeled = FilterBuilder.build_labeled_clauses(
        _resolved(),
        QuerySpec(cdf_threshold=150, cdf_threshold_channel_localizations=["membrane"]),
    )
    labels = [label for label, _ in labeled]
    assert any("resolved for 'membrane'" in label for label in labels)
    assert any("cdf_90>=150.0" in label for label in labels)


def test_cdf_threshold_rejected_on_tile_view():
    with pytest.raises(NotImplementedError, match="tiles_training"):
        FilterBuilder.build_where_sql(
            _resolved(SampleType.TILE),
            QuerySpec(cdf_threshold=150, cdf_threshold_channel_localizations=["membrane"]),
        )


def test_complete_tiles_only_is_cube_only():
    """is_complete is projected on cube_training only; tiles_training filters
    windows overlapping missing_timepoints internally."""
    assert "is_complete" in FilterBuilder.build_where_sql(
        _resolved(SampleType.CUBE), QuerySpec(complete_tiles_only=True)
    )
    assert "is_complete" not in FilterBuilder.build_where_sql(
        _resolved(SampleType.TILE), QuerySpec(complete_tiles_only=True)
    )


def test_missing_channel_metadata_diagnostic_uses_the_catalog():
    """Asked the same way the filters are: a row is dropped by every channel
    predicate when its ROI has no api.roi_channels rows at all."""
    resolved = _resolved()
    sql = SqlQueryPlanner.build_missing_channel_metadata_diagnostic_sql(
        resolved, QuerySpec(required_channel_localizations=["membrane"])
    )
    assert "NOT EXISTS (SELECT 1 FROM api.roi_channels rc" in sql
    assert SqlQueryPlanner.build_missing_channel_metadata_diagnostic_sql(
        resolved, QuerySpec()
    ) is None


def test_build_sql_appends_limit_when_max_rows_set():
    assert "LIMIT 5" in SqlQueryPlanner.build_sql(_resolved(), QuerySpec(max_rows=5))


# --------------------------------------------------------------------------- #
# channel roles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "channel_type,annotation_type,expected",
    [
        ("mask", "instance", "instance_masks"),
        ("mask", "semantic", "semantic_masks"),
        ("data", None, None),
    ],
)
def test_channel_role_from_type_and_annotation(channel_type, annotation_type, expected):
    """Two DB columns answer "is it GT" and "which GT" separately, and the answer
    IS DataKind -- so the 5-entry _OBJECT_FAMILIES hardcode is gone."""
    assert channel_role(channel_type, annotation_type) == expected


def test_channel_role_requires_annotation_type_on_mask():
    with pytest.raises(ValueError, match="annotation_type"):
        channel_role("mask", None)


def test_channel_role_values_are_object_roles():
    """Pins the two vocabularies together: every role we mint must parse as a
    GT DataKind."""
    assert is_object_role(channel_role("mask", "instance"))
    assert is_object_role(channel_role("mask", "semantic"))


def test_unknown_role_is_not_an_object_role():
    """A data channel may carry a biology token ("membrane") as its role, so an
    unknown token must be False rather than an error -- partition_channels asks
    this of every roled channel. The dangerous case (a role matching a declared
    TARGET family but not an object role) is caught in partition_channels, where
    both pieces of information are in hand."""
    assert is_object_role("membrane") is False
    assert is_object_role("not_a_kind") is False


# --------------------------------------------------------------------------- #
# sampling (unchanged behaviour, kept as regression cover)
# --------------------------------------------------------------------------- #


def test_split_train_val_is_deterministic_and_disjoint():
    train, val = SampleIndexPlanner.split_train_val(100, 0.2, seed=0)
    assert len(train) == 80 and len(val) == 20
    assert not set(train.tolist()) & set(val.tolist())
    again, _ = SampleIndexPlanner.split_train_val(100, 0.2, seed=0)
    assert np.array_equal(train, again)


def test_plan_epoch_shards_by_rank_without_overlap():
    base = np.arange(16, dtype=np.int64)
    r0 = SampleIndexPlanner(world_size=2, rank=0).plan_epoch(
        base, seed=0, shuffle=False, batch_size=2, last_batch_policy="drop"
    )
    r1 = SampleIndexPlanner(world_size=2, rank=1).plan_epoch(
        base, seed=0, shuffle=False, batch_size=2, last_batch_policy="drop"
    )
    assert not set(r0.tolist()) & set(r1.tolist())


def test_server_path_override_replaces_storage_root():
    """The escape hatch that survived _remap_server_folder's deletion: it runs
    only when a config asks, and overrides a value the DB otherwise owns."""
    table = pa.table({"storage_root": pa.array(["/catalog/root", "/catalog/root"])})
    out = MappedTable._apply_server_path_override(table, "/local/mount")
    assert out["storage_root"].to_pylist() == ["/local/mount", "/local/mount"]
    assert MappedTable._apply_server_path_override(table, None) is table


# --------------------------------------------------------------------------- #
# temporal-extent guard
# --------------------------------------------------------------------------- #


def _time_table(*sizes: int) -> pa.Table:
    return pa.table({"time_size": pa.array(list(sizes), type=pa.int64())})


def test_time_size_guard_accepts_a_match():
    _assert_requested_time_size(_time_table(1, 1), _resolved())


def test_time_size_guard_rejects_a_broadcast():
    """The tile case: no shape predicate is emitted, so a T=16 request comes back
    as time_size=1 rows and the 4D pad branch would broadcast frame 0."""
    resolved = ResolvedSource(
        view=TRAINING_VIEWS[SampleType.TILE], requested_time_size=16,
        requested_z_size=64, requested_y_size=64, requested_x_size=64,
        with_targets=False, location_id=3, location_name="synthetic",
    )
    with pytest.raises(ValueError, match="requested time_size=16"):
        _assert_requested_time_size(_time_table(1, 1), resolved)


def test_time_size_guard_raises_without_the_column():
    """time_size is in schema.LOADER_COLUMNS, so validate_projection would have
    caught this first -- but a caller that skips validation must not silently
    lose the guard."""
    with pytest.raises(ValueError, match="no time_size column"):
        _assert_requested_time_size(pa.table({"row_id": pa.array([0])}), _resolved())


def test_time_size_guard_is_a_noop_on_an_empty_result():
    """Any filter may legitimately select nothing, and an empty result is already
    reported by the narrowing diagnostic -- which for a CUBE T mismatch names the
    `time_size=N` step, because cubes pin T in the WHERE."""
    _assert_requested_time_size(_time_table(), _resolved())


# --------------------------------------------------------------------------- #
# host-buffer channel sizing
# --------------------------------------------------------------------------- #


def _channel_table(n_rois: int = 2) -> pa.Table:
    """Rows sharing the real ROI layout: membrane, 4x cytosol, 1 instance mask."""
    return pa.table({
        "roi_id": pa.array([1] * 3 + [2] * 3, type=pa.int64()),
        "channel_idx": pa.array([[0, 1, 2, 3, 4, 5]] * 6),
        "channel_type": pa.array([["data"] * 5 + ["mask"]] * 6),
        "localization": pa.array(
            [["membrane", "cytosol", "cytosol", "cytosol", "cytosol", None]] * 6
        ),
    })


@pytest.mark.parametrize(
    "selection,expected",
    [
        (["membrane"], 2),              # 1 data + 1 mask
        (["cytosol"], 5),               # 4 data + 1 mask
        (["membrane", "cytosol"], 6),   # 5 data + 1 mask
    ],
)
def test_selected_channel_size_counts_emitted_channels(selection, expected):
    """NOT len(selection). One localization can match several channels (cytosol
    matches four here) and the mask channels are always retained, so the token
    count undersizes the host buffer and the loader writes past it.

    The old sizing happened to be right only because the previous dataset had one
    channel per localization and no retained mask.
    """
    assert _selected_channel_size(_channel_table(), selection) == expected
    assert _selected_channel_size(_channel_table(), selection) != len(selection)


def test_selected_channel_size_is_zero_without_a_selection():
    """No selection -> every channel is loaded, so max_channel_size is the answer
    and _shape_from_stats keeps using it."""
    assert _selected_channel_size(_channel_table(), None) == 0
    assert _selected_channel_size(_channel_table(), []) == 0


# --------------------------------------------------------------------------- #
# channel vocabulary validation
# --------------------------------------------------------------------------- #


class _FakeDb:
    """Returns the api.roi_channels shape the real DB has."""

    def __init__(self, rows=(("membrane", "Electra2"), ("cytosol", "mTFP1"), (None, None))):
        self.rows = rows
        self.sql = None

    def execute_arrow(self, sql):
        self.sql = sql
        return pa.table({
            "localization": pa.array([r[0] for r in self.rows]),
            "fluorophore": pa.array([r[1] for r in self.rows]),
        })


def test_vocabulary_comes_from_the_data_not_a_terms_table():
    """wet_lab.localization_terms is the nominal authority but is under-seeded --
    no 'membrane', which every config asks for -- so validating against it would
    reject every run. api.roi_channels has no such gap by construction, and it is
    the same column the filters and the loader match against."""
    db = _FakeDb()
    TableResolver._fetch_channel_vocabulary(db)
    assert CHANNEL_CATALOG in db.sql
    assert "localization_terms" not in db.sql
    assert "channel_type = 'data'" in db.sql


def test_vocabulary_is_normalized_and_drops_nulls():
    """fluorophore is mixed-case in the DB ("Electra2", "mTFP1") and both columns
    are NULL on mask channels."""
    vocab = TableResolver._fetch_channel_vocabulary(_FakeDb())
    assert vocab["localization"] == {"membrane", "cytosol"}
    assert vocab["fluorophore"] == {"electra2", "mtfp1"}


@pytest.mark.parametrize("column,requested", [
    ("localization", ["membrane"]),
    ("localization", ["MEMBRANE"]),
    ("fluorophore", ["Electra2"]),
    ("fluorophore", ["electra2"]),
])
def test_known_tokens_accepted_case_insensitively(column, requested):
    TableResolver._assert_known_tokens(_FakeDb(), column, requested)


def test_unknown_token_raises_and_lists_what_exists():
    """Without this the same typo is either a silently empty dataset or a
    ValueError raised deep inside a Ray loader actor, long after startup."""
    with pytest.raises(ValueError, match=r"golgi"):
        TableResolver._assert_known_tokens(_FakeDb(), "localization", ["membrane", "golgi"])


def test_validation_is_skipped_without_a_db_client():
    """Offline config construction must not require a connection."""
    TableResolver._assert_known_tokens(None, "localization", ["anything"])


# --------------------------------------------------------------------------- #
# seeded row sampling / tile shape bound
# --------------------------------------------------------------------------- #

def test_seeded_row_sample_ranks_by_md5_and_restores_natural_order():
    sql = SqlQueryPlanner.build_sql(_resolved(), QuerySpec(max_rows=3, row_sample_seed=7))
    compact = " ".join(sql.split())
    assert "md5(concat_ws('/', s.source_kind::text" in compact and "/7')" in compact
    assert "LIMIT 3" in compact
    # outer query: row_id over the natural key of the sampled subset, natural order restored
    assert "row_number() OVER (ORDER BY q.source_kind, q.roi_id" in compact
    assert compact.rstrip().endswith("ORDER BY q.source_kind, q.roi_id, q.tile_id, q.time_start, q.z_start, q.y_start, q.x_start")


def test_unseeded_max_rows_keeps_natural_order_limit():
    sql = SqlQueryPlanner.build_sql(_resolved(), QuerySpec(max_rows=3))
    assert "md5(" not in sql and sql.rstrip().endswith("LIMIT 3")


def test_max_tile_shape_bounds_each_axis_on_tiles_only():
    where = FilterBuilder.build_where_sql(_resolved(SampleType.TILE), QuerySpec(max_tile_shape=(128, 512, 3328)))
    assert "s.z_size <= 128" in where and "s.y_size <= 512" in where and "s.x_size <= 3328" in where
    where_cube = FilterBuilder.build_where_sql(_resolved(SampleType.CUBE), QuerySpec(max_tile_shape=(128, 512, 3328)))
    assert "s.z_size <= 128" not in where_cube and "s.x_size <= 3328" not in where_cube


def test_out_of_bounds_cubes_raise_unless_dropping_is_opted_in():
    from cell_observatory_platform.data.databases.local_metadata_store import validate_cube_bounds
    table = pa.table({
        "z_start": [0, 0], "y_start": [0, 384], "x_start": [0, 0],
        "z_size": [128, 128], "y_size": [384, 384], "x_size": [384, 384],
        "array_shape": [[1, 128, 512, 3328, 6], [1, 128, 512, 3328, 6]],
    })
    with pytest.raises(ValueError, match=r"1 of 2 cubes extend past.*128x384x384"):
        validate_cube_bounds(table, _resolved(SampleType.CUBE), QuerySpec(), where="t")
    validate_cube_bounds(table, _resolved(SampleType.CUBE), QuerySpec(in_bounds_only=True), where="t")  # warns only
    validate_cube_bounds(table, _resolved(SampleType.TILE), QuerySpec(), where="t")                     # tiles: no-op
    ok = table.slice(0, 1)
    validate_cube_bounds(ok, _resolved(SampleType.CUBE), QuerySpec(), where="t")


def test_in_bounds_clause_only_when_opted_in():
    assert "array_shape" not in FilterBuilder.build_where_sql(_resolved(SampleType.CUBE), QuerySpec())
    assert "s.y_start + s.y_size <= s.array_shape[3]" in FilterBuilder.build_where_sql(
        _resolved(SampleType.CUBE), QuerySpec(in_bounds_only=True))
