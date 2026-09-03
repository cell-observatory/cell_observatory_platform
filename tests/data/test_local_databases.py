"""Live-DB checks against the api training views (needs --run-localdb).

Deliberately small: the column contract is owned by ``data/databases/schema.py``
and pinned by the offline suite, and shape is a predicate rather than part of a
relation name, so there is nothing here to enumerate per relation.

What is left is only what CANNOT be checked offline -- the things that need a
real server:

  * the projection the DB actually returns matches ``required_columns``
  * the natural key is genuinely unique (resume identity rests on it)
  * ``root_path || tile_relative_path`` resolves to something on disk
  * the api.roi_channels semi-joins and the CDF subscript are accepted by
    Postgres and narrow rather than error

Everything else -- clause text, ordering, projection shape -- is asserted in
tests/data/test_local_metadata_store.py without a database.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cell_observatory_platform.data.databases.local_database import LocalArrowDatabase
from cell_observatory_platform.data.databases.local_metadata_store import (
    TRAINING_VIEWS,
    FilterBuilder,
    MappedTable,
    QuerySpec,
    SampleType,
    SqlQueryPlanner,
    TableResolver,
    fetch_object_type_names,
    fetch_storage_locations,
)
from cell_observatory_platform.data.databases.schema import (
    required_columns,
    validate_non_null,
    validate_projection,
)

pytestmark = pytest.mark.localdb


@pytest.fixture(autouse=True)
def _disable_barrier(monkeypatch):
    monkeypatch.setattr(
        "cell_observatory_platform.data.databases.local_metadata_store.barrier",
        lambda *a, **k: None,
    )


@pytest.fixture(scope="session")
def local_db() -> LocalArrowDatabase:
    """Connect to whatever DB the operator points us at.

    COD_DATABASE_DSN (the env var the cell_observatory_database repo's own tests
    use) wins, so a restored dump on an arbitrary host/port can be tested without
    editing .env. Falling back to dbname="local" keeps the original behaviour,
    which resolves the host from node_ip() + SUPABASE_LOCAL_PORT.
    """
    dsn = os.environ.get("COD_DATABASE_DSN") or os.environ.get("PG17_DSN")
    if dsn:
        return LocalArrowDatabase(dbname="local", database_url=dsn)
    return LocalArrowDatabase(dbname="local", dotenv_path=os.environ.get("DOTENV_PATH"))


def _populated_location(local_db, view) -> tuple[str, int]:
    """The storage location most rows actually live at.

    NOT just the first name: the catalog lists several locations (vast-main,
    abc2-main, synthetic) but a given export populates one of them, and the
    location join is an INNER join -- picking an empty one yields zero rows, and
    every downstream assertion then degrades into a skip, which reads like a pass.
    """
    locations = fetch_storage_locations(local_db)
    assert locations, "dry_lab.storage_locations has no active rows"
    by_id = {v: k for k, v in locations.items()}
    rows = local_db.execute_arrow(
        f"SELECT unnest(present_locations) AS loc, count(*) AS n "
        f"FROM {view.qualified_name} GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    )
    assert rows.num_rows, f"{view.qualified_name} has no rows at any storage location"
    loc_id = int(rows["loc"][0].as_py())
    return by_id[loc_id], loc_id


def _resolved(local_db, sample_type: SampleType, *, with_targets: bool = False):
    view = TRAINING_VIEWS[sample_type]
    name, loc_id = _populated_location(local_db, view)
    from cell_observatory_platform.data.databases.local_metadata_store import ResolvedSource

    return ResolvedSource(
        view=view,
        requested_time_size=1,
        requested_z_size=128,
        requested_y_size=128,
        requested_x_size=128,
        with_targets=with_targets,
        location_id=loc_id,
        location_name=name,
    )


def _build_config(tmp_path: Path, sample_type: SampleType, location_name: str):
    return OmegaConf.create(
        {
            "dataset_layout_order": "ZYXC",
            "datasets": {
                "input_shape": [128, 128, 128, 2],
                "has_annotations": False,
                "roi_ids": None,
                "tile_list": None,
                "timepoint_list": None,
                "synthetic_only": False,
                "cdf_threshold": None,
                "cdf_target": "90",
                "selected_channel_localizations": None,
                "databases": {
                    "sample_type": sample_type.value,
                    "storage_location": location_name,
                    "node_local_store_root": str(tmp_path),
                },
            },
        }
    )


# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sample_type", [SampleType.CUBE, SampleType.TILE])
def test_view_exists_and_projects_the_contract(local_db, sample_type):
    """The one check that can only be made against a live server: does the view
    actually return every column data/databases/schema.py claims it does."""
    view = TRAINING_VIEWS[sample_type]
    local_db.assert_relation_exists(view.view, schema=view.schema)

    resolved = _resolved(local_db, sample_type, with_targets=True)
    sql = SqlQueryPlanner.build_sql(resolved, QuerySpec(max_rows=0))
    table = local_db.execute_arrow(sql)

    validate_projection(
        table,
        with_targets=True,
        cube=view.is_cube,
        where=f"{view.qualified_name} live projection",
    )
    assert "row_id" in table.column_names


@pytest.mark.parametrize("sample_type", [SampleType.CUBE, SampleType.TILE])
def test_rows_are_loadable(local_db, sample_type):
    """No NULL in anything the loader dereferences unconditionally."""
    resolved = _resolved(local_db, sample_type)
    table = local_db.execute_arrow(
        SqlQueryPlanner.build_sql(resolved, QuerySpec(max_rows=512))
    )
    if table.num_rows == 0:
        pytest.skip(f"{resolved.view.qualified_name} returned no rows for this shape")
    validate_non_null(table, where=f"{resolved.view.qualified_name} live rows")


def test_natural_key_is_unique_on_cube_view(local_db):
    """Resume identity. row_id comes from row_number() OVER (ORDER BY natural
    key), and skip_batches replays a row RANGE -- a duplicate key means a resumed
    run silently trains on different data.

    Asserted server-side (count(*) vs count(DISTINCT ...)) rather than by pulling
    5M rows into Arrow.
    """
    view = TRAINING_VIEWS[SampleType.CUBE]
    cols = ", ".join(view.order_by)
    row = local_db.execute_arrow(
        f"SELECT count(*) AS n, count(DISTINCT ({cols})) AS d FROM {view.qualified_name}"
    )
    n, d = int(row["n"][0].as_py()), int(row["d"][0].as_py())
    assert n == d, (
        f"{view.qualified_name}: ORDER BY key {view.order_by} is not unique "
        f"({n} rows, {d} distinct)"
    )


def test_storage_root_join_resolves_on_disk(local_db):
    """root_path || '/' || tile_relative_path must be a real path -- and must NOT
    have tile_name joined on again (tile_relative_path already ends in the tile).
    """
    resolved = _resolved(local_db, SampleType.CUBE)
    table = local_db.execute_arrow(
        SqlQueryPlanner.build_sql(resolved, QuerySpec(max_rows=5))
    )
    if table.num_rows == 0:
        pytest.skip("no cube rows for this shape/location")
    for root, rel, name in zip(
        table["storage_root"].to_pylist(),
        table["tile_relative_path"].to_pylist(),
        table["tile_name"].to_pylist(),
    ):
        assert rel.endswith(name), "tile_relative_path should already end in the tile"
        assert os.path.exists(os.path.join(root, rel)), f"missing: {root}/{rel}"


def test_channel_and_cdf_filters_are_accepted_and_narrow(local_db):
    """The api.roi_channels semi-joins and the occupancy_cdf subscript have to be
    valid SQL against the real column types, and should narrow rather than error.

    Worth running live because the semi-join joins two api views on roi_id and
    the CDF subscript indexes an array whose mask slots are NULL -- neither is
    checkable from the generated string alone.
    """
    resolved = _resolved(local_db, SampleType.CUBE)
    baseline = local_db.execute_arrow(
        f"SELECT count(*) AS n FROM {resolved.view.qualified_name}"
    )["n"][0].as_py()

    for query in (
        QuerySpec(required_channel_localizations=["membrane"]),
        QuerySpec(required_fluorophores=["mstaygold"]),
        QuerySpec(required_annotation_types=["instance"]),
        QuerySpec(
            cdf_threshold=1,
            cdf_threshold_channel_localizations=["membrane"],
        ),
    ):
        where = FilterBuilder.build_where_sql(resolved, query)
        got = local_db.execute_arrow(
            f"SELECT count(*) AS n FROM {resolved.view.qualified_name} s WHERE {where}"
        )["n"][0].as_py()
        assert 0 <= int(got) <= int(baseline)


def test_object_type_catalog_resolves_names(local_db):
    """Annotation leaves carry object_type_id only; api.object_types is the
    catalog configs' class names resolve through."""
    names = fetch_object_type_names(local_db)
    assert names, "api.object_types is empty; semantic class names cannot resolve"
    assert all(isinstance(k, int) and isinstance(v, str) for k, v in names.items())


def test_resolver_and_materialization_round_trip(local_db, tmp_path):
    name, loc_id = _populated_location(local_db, TRAINING_VIEWS[SampleType.CUBE])
    cfg = _build_config(tmp_path, SampleType.CUBE, name)

    resolved = TableResolver.resolve_from_config(cfg, db_client=local_db)
    assert resolved.view.qualified_name == "api.cube_training"
    assert resolved.location_id == loc_id

    query = TableResolver.build_query_spec_from_config(cfg)
    store = TableResolver.build_store_spec_from_config(cfg)

    mapped = MappedTable.create_or_attach(
        db_client=local_db,
        resolved=resolved,
        query=query,
        store=store,
        node_id="test",
        local_rank=0,
    )
    stats = mapped.descriptor.stats
    assert stats.num_rows >= 0
    assert stats.fixed_axes == ("T", "Z", "Y", "X")
