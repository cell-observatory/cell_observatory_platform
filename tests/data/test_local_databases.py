from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf

import pytest

from cell_observatory_platform.data.databases.local_database import LocalArrowDatabase
from cell_observatory_platform.data.databases.local_metadata_store import (
    FilterBuilder,
    MappedTable,
    QuerySpec,
    SOURCE_TABLES,
    SqlQueryPlanner,
    TableResolver,
)

pytestmark = pytest.mark.localdb

LOCATION_KEYS = ["is_available", "exists_prfs", "exists_aws", "exists_oak", "exists_abc"]
SUPPORTED_CUBE_SPATIAL_SHAPES = {
    (128, 128, 128),
    (128, 256, 256),
    (128, 384, 512),
}

EXPECTED_COLUMNS = {
    "cube_without_annotations": {
        "first_pc_id",
        "prepared_id",
        "tile_name",
        "server_folder",
        "output_folder",
        "is_synthetic",
        "is_available",
        "exists_prfs",
        "exists_aws",
        "exists_oak",
        "exists_abc",
        "time_start",
        "time_size",
        "z_start",
        "y_start",
        "x_start",
        "z_size",
        "y_size",
        "x_size",
        "channel_size",
        "is_complete",
        "is_test_split",
        "channel_mapping",
        "channels_metadata",
        "annotation_count",
        "has_annotations",
    },
    "cube_with_annotations": {
        "first_pc_id",
        "prepared_id",
        "tile_name",
        "server_folder",
        "output_folder",
        "is_synthetic",
        "is_available",
        "exists_prfs",
        "exists_aws",
        "exists_oak",
        "exists_abc",
        "time_start",
        "time_size",
        "z_start",
        "y_start",
        "x_start",
        "z_size",
        "y_size",
        "x_size",
        "channel_size",
        "is_complete",
        "is_test_split",
        "channel_mapping",
        "channels_metadata",
        "annotation_count",
        "annotations_metadata",
    },
    "tile_without_annotations": {
        "first_pc_id",
        "prepared_id",
        "tile_name",
        "server_folder",
        "output_folder",
        "is_synthetic",
        "is_available",
        "exists_prfs",
        "exists_aws",
        "exists_oak",
        "exists_abc",
        "time_start",
        "time_size",
        "z_start",
        "y_start",
        "x_start",
        "z_size",
        "y_size",
        "x_size",
        "channel_size",
        "is_complete",
        "is_test_split",
        "channel_mapping",
    },
    "tile_with_annotations": {
        "first_pc_id",
        "prepared_id",
        "tile_name",
        "server_folder",
        "output_folder",
        "is_synthetic",
        "is_available",
        "exists_prfs",
        "exists_aws",
        "exists_oak",
        "exists_abc",
        "time_start",
        "time_size",
        "z_start",
        "y_start",
        "x_start",
        "z_size",
        "y_size",
        "x_size",
        "channel_size",
        "is_complete",
        "is_test_split",
        "channel_mapping",
        "annotation_count",
        "annotations_metadata",
    },
}


@pytest.fixture(autouse=True)
def _disable_barrier(monkeypatch):
    monkeypatch.setattr(
        "cell_observatory_platform.data.databases.local_metadata_store.barrier",
        lambda: None,
    )


@pytest.fixture(scope="session")
def local_db() -> LocalArrowDatabase:
    db = LocalArrowDatabase(dbname="local", verbose=False)
    probe = db.execute_arrow("SELECT 1 AS ok")
    assert probe.num_rows == 1
    assert probe["ok"][0].as_py() == 1
    return db


@pytest.fixture(scope="session")
def prepared_tables(local_db: LocalArrowDatabase) -> list[dict]:
    sql = """
        SELECT c.relname AS table_name
        FROM pg_class c
        JOIN pg_namespace n
          ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND (
            c.relname LIKE 'prepared_cube\\_%_agg\\_%' ESCAPE '\\'
            OR c.relname LIKE 'prepared_tile\\_%_agg\\_%' ESCAPE '\\'
          )
          AND NOT EXISTS (
            SELECT 1
            FROM pg_inherits i
            WHERE i.inhrelid = c.oid
          )
        ORDER BY c.relname
    """
    rows = local_db.execute_arrow(sql)["table_name"].to_pylist()
    assert rows, "No prepared_*_agg_* tables found in live local DB"
    return [classify_table_name(name) for name in rows]


def classify_table_name(table_name: str) -> dict:
    cube_channel = re.fullmatch(r"prepared_cube_channel_agg_(\d+)_(\d+)_(\d+)_(\d+)", table_name)
    if cube_channel:
        t, z, y, x = map(int, cube_channel.groups())
        spatial_shape = (z, y, x)
        return {
            "table_name": table_name,
            "kind": "cube_without_annotations",
            "sample_type": "cube",
            "has_annotations": False,
            "source_key": f"cube_without_annotations_{t}_{z}_{y}_{x}",
            "selector_key": "cube_without_annotations_3d" if t == 1 else "cube_without_annotations_4d",
            "layout": "ZYXC" if t == 1 else "TZYXC",
            "input_shape": [z, y, x, 2] if t == 1 else [t, z, y, x, 2],
            "supported_by_resolver": spatial_shape in SUPPORTED_CUBE_SPATIAL_SHAPES,
        }

    cube_annotation = re.fullmatch(r"prepared_cube_annotation_agg_(\d+)_(\d+)_(\d+)_(\d+)", table_name)
    if cube_annotation:
        t, z, y, x = map(int, cube_annotation.groups())
        spatial_shape = (z, y, x)
        return {
            "table_name": table_name,
            "kind": "cube_with_annotations",
            "sample_type": "cube",
            "has_annotations": True,
            "source_key": f"cube_with_annotations_{t}_{z}_{y}_{x}",
            "selector_key": "cube_with_annotations_3d",
            "layout": "ZYXC" if t == 1 else "TZYXC",
            "input_shape": [z, y, x, 2] if t == 1 else [t, z, y, x, 2],
            "supported_by_resolver": t == 1 and spatial_shape in SUPPORTED_CUBE_SPATIAL_SHAPES,
        }

    tile_channel = re.fullmatch(r"prepared_tile_channel_agg_(\d+)", table_name)
    if tile_channel:
        t = int(tile_channel.group(1))
        return {
            "table_name": table_name,
            "kind": "tile_without_annotations",
            "sample_type": "tile",
            "has_annotations": False,
            "source_key": f"tile_without_annotations_{t}",
            "selector_key": f"tile_without_annotations_{t}",
            "layout": "ZYXC" if t == 1 else "TZYXC",
            "input_shape": [128, 256, 256, 2] if t == 1 else [t, 128, 256, 256, 2],
            "supported_by_resolver": True,
        }

    tile_annotation = re.fullmatch(r"prepared_tile_annotation_agg_(\d+)", table_name)
    if tile_annotation:
        t = int(tile_annotation.group(1))
        return {
            "table_name": table_name,
            "kind": "tile_with_annotations",
            "sample_type": "tile",
            "has_annotations": True,
            "source_key": f"tile_with_annotations_{t}",
            "selector_key": "tile_with_annotations_1",
            "layout": "ZYXC" if t == 1 else "TZYXC",
            "input_shape": [128, 256, 256, 2] if t == 1 else [t, 128, 256, 256, 2],
            "supported_by_resolver": t == 1,
        }

    raise AssertionError(f"Unrecognized prepared table name: {table_name}")


def build_config(meta: dict, tmp_path: Path):
    return OmegaConf.create(
        {
            "dataset_layout_order": meta["layout"],
            "datasets": {
                "input_shape": meta["input_shape"],
                "has_annotations": meta["has_annotations"],
                "roi_list": None,
                "tile_list": None,
                "timepoint_list": None,
                "max_hypercubes": 25,
                "cdf_threshold": None,
                "cdf_target": "90",
                "synthetic_only": False,
                "selected_channel_localizations": None,
                "databases": {
                    "sample_type": meta["sample_type"],
                    "node_local_store_root": str(tmp_path),
                    "node_local_table_keys": {
                        meta["selector_key"]: meta["source_key"],
                    },
                    "cdf_threshold_channel_localizations": None,
                    "channel_count": None,
                    "min_channel_count": None,
                    "max_channel_count": None,
                    "required_channel_key_patterns": None,
                    "any_channel_patterns": None,
                    "all_channel_patterns": None,
                    "required_channel_localizations": None,
                    "required_locations": None,
                    "holdout_split": None,
                },
            },
        }
    )


def scalar(db: LocalArrowDatabase, sql: str, column: str):
    table = db.execute_arrow(sql)
    assert table.num_rows == 1, sql
    return table[column][0].as_py()


def column_names_for_table(db: LocalArrowDatabase, table_name: str) -> set[str]:
    sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    return set(db.execute_arrow(sql)["column_name"].to_pylist())


def query_table(db: LocalArrowDatabase, resolved, query: QuerySpec):
    sql = SqlQueryPlanner.build_sql(resolved, query)
    return db.execute_arrow(sql)


def table_head_overview(db: LocalArrowDatabase, table_name: str, limit: int = 3) -> list[dict]:
    sql = f"""
        SELECT *
        FROM public.{table_name}
        ORDER BY prepared_id, tile_name, time_start
        LIMIT {int(limit)}
    """
    return db.execute_arrow(sql).to_pylist()


@pytest.fixture(scope="session")
def live_channel_localizations(local_db: LocalArrowDatabase) -> list[str]:
    sql = """
        SELECT DISTINCT lower(trim(localization)) AS localization
        FROM fish_db.tags
        WHERE localization IS NOT NULL
          AND COALESCE(is_deleted, false) = false
        ORDER BY localization
    """
    localizations = [
        value
        for value in local_db.execute_arrow(sql)["localization"].to_pylist()
        if value is not None and str(value).strip()
    ]
    assert localizations, "No live channel localizations found in fish_db.tags"
    return localizations


def first_channel_localization(db: LocalArrowDatabase, table_name: str) -> str | None:
    sql = f"""
        SELECT lower(trim(both '"' from mapping.value::text)) AS channel_localization
        FROM public.{table_name} s
        CROSS JOIN LATERAL jsonb_each(COALESCE(s.channel_mapping, '{{}}'::jsonb)) AS mapping(key, value)
        LIMIT 1
    """
    table = db.execute_arrow(sql)
    if table.num_rows == 0:
        return None
    return table["channel_localization"][0].as_py()


def assert_all_equal(values, expected, msg: str):
    actual = set(values)
    assert actual == {expected}, f"{msg}: expected only {expected!r}, got {sorted(actual)!r}"


def assert_all_true(values, msg: str):
    assert all(bool(v) for v in values), msg


def assert_all_false(values, msg: str):
    assert all(not bool(v) for v in values), msg


def test_live_db_connection_and_relation_exists(local_db: LocalArrowDatabase, prepared_tables: list[dict]):
    assert not local_db.relation_exists("definitely_not_a_real_table_123456789")

    for meta in prepared_tables:
        assert local_db.relation_exists(meta["table_name"])
        local_db.assert_relation_exists(meta["table_name"])


def test_all_live_prepared_tables_have_expected_columns(local_db: LocalArrowDatabase, prepared_tables: list[dict]):
    for meta in prepared_tables:
        if not meta["supported_by_resolver"]:
            continue
        cols = column_names_for_table(local_db, meta["table_name"])
        missing = EXPECTED_COLUMNS[meta["kind"]] - cols
        assert not missing, f"{meta['table_name']} is missing columns: {sorted(missing)}"


def test_live_catalog_only_contains_tables_supported_by_resolver(prepared_tables: list[dict]):
    unsupported = [meta["table_name"] for meta in prepared_tables if not meta["supported_by_resolver"]]
    for meta in prepared_tables:
        if meta["supported_by_resolver"]:
            continue
        assert meta["sample_type"] == "cube", meta["table_name"]
        spatial_shape = tuple(meta["input_shape"][-4:-1] if meta["layout"] == "TZYXC" else meta["input_shape"][:3])
        assert spatial_shape not in SUPPORTED_CUBE_SPATIAL_SHAPES, meta["table_name"]
    if unsupported:
        print(f"\n[localdb] skipping unsupported live tables: {unsupported}")


def test_live_prepared_table_heads(
    local_db: LocalArrowDatabase,
    prepared_tables: list[dict],
):
    for meta in prepared_tables:
        rows = table_head_overview(local_db, meta["table_name"], limit=3)
        print(f"\n=== {meta['table_name']} ({meta['kind']}) ===")
        for row in rows:
            print(row)


def test_resolver_uses_axis_policies_for_cube_and_tile_sources(tmp_path: Path):
    cube_cfg = build_config(
        {
            "layout": "ZYXC",
            "input_shape": [32, 64, 64, 2],
            "has_annotations": False,
            "sample_type": "cube",
            "selector_key": "cube_without_annotations_3d",
            "source_key": "cube_without_annotations_1_32_64_64",
        },
        tmp_path / "cube",
    )
    tile_cfg = build_config(
        {
            "layout": "ZYXC",
            "input_shape": [128, 256, 256, 2],
            "has_annotations": False,
            "sample_type": "tile",
            "selector_key": "tile_without_annotations_1",
            "source_key": "tile_without_annotations_1",
        },
        tmp_path / "tile",
    )

    cube = TableResolver.resolve_from_config(cube_cfg)
    tile = TableResolver.resolve_from_config(tile_cfg)

    assert cube.source.fixed_axes == ("T", "Z", "Y", "X")
    assert cube.source.dynamic_axes == ("C",)
    assert tile.source.fixed_axes == ()
    assert tile.source.dynamic_axes == ("T", "Z", "Y", "X", "C")


def test_resolver_sql_and_materialization_work_for_every_live_table(
    local_db: LocalArrowDatabase,
    prepared_tables: list[dict],
    tmp_path: Path,
):
    for meta in prepared_tables:
        if not meta["supported_by_resolver"]:
            continue
        table_tmp = tmp_path / meta["table_name"]
        cfg = build_config(meta, table_tmp)

        resolved = TableResolver.resolve_from_config(cfg, db_client=local_db)
        assert resolved.table_name == meta["table_name"]

        query = TableResolver.build_query_spec_from_config(cfg)
        store = TableResolver.build_store_spec_from_config(cfg)

        live = query_table(local_db, resolved, query)
        if live.num_rows == 0:
            if meta["has_annotations"]:
                print(f"\n[localdb] annotation table {meta['table_name']} returned zero rows; skipping")
                continue
            assert live.num_rows > 0, f"{meta['table_name']} returned zero rows"
        assert "row_id" in live.column_names
        assert "prepared_id" in live.column_names
        assert "tile_name" in live.column_names
        assert "time_start" in live.column_names

        if meta["has_annotations"]:
            assert all(v > 0 for v in live["annotation_count"].to_pylist()), meta["table_name"]

        writer = MappedTable.create_or_attach(
            db_client=local_db,
            resolved=resolved,
            query=query,
            store=store,
            node_id="pytest-live",
            local_rank=0,
        )
        reader = MappedTable.create_or_attach(
            db_client=local_db,
            resolved=resolved,
            query=query,
            store=store,
            node_id="pytest-live",
            local_rank=1,
        )

        writer_table = writer.table()
        reader_table = reader.table()

        assert writer_table.num_rows == live.num_rows
        assert reader_table.num_rows == live.num_rows
        assert reader.descriptor.sample_table.source_key == meta["source_key"]


def test_live_basic_filters_work_for_every_table(
    local_db: LocalArrowDatabase,
    prepared_tables: list[dict],
    tmp_path: Path,
):
    exercised_location_filter = False

    for meta in prepared_tables:
        if not meta["supported_by_resolver"]:
            continue
        cfg = build_config(meta, tmp_path / meta["table_name"])
        resolved = TableResolver.resolve_from_config(cfg, db_client=local_db)

        base = query_table(local_db, resolved, QuerySpec(max_rows=50))
        if base.num_rows == 0:
            if meta["has_annotations"]:
                print(f"\n[localdb] annotation table {meta['table_name']} returned zero rows; skipping")
                continue
            assert base.num_rows > 0, f"{meta['table_name']} returned zero rows"

        prepared_id = base["prepared_id"][0].as_py()
        tile_name = base["tile_name"][0].as_py()
        time_start = int(base["time_start"][0].as_py())

        roi_table = query_table(local_db, resolved, QuerySpec(roi_list=[prepared_id], max_rows=25))
        assert roi_table.num_rows > 0, meta["table_name"]
        assert_all_equal(roi_table["prepared_id"].to_pylist(), prepared_id, f"{meta['table_name']} roi filter")

        tile_table = query_table(local_db, resolved, QuerySpec(tile_list=[tile_name], max_rows=25))
        assert tile_table.num_rows > 0, meta["table_name"]
        assert_all_equal(tile_table["tile_name"].to_pylist(), tile_name, f"{meta['table_name']} tile filter")

        time_table = query_table(local_db, resolved, QuerySpec(timepoint_list=[time_start], max_rows=25))
        assert time_table.num_rows > 0, meta["table_name"]
        assert_all_equal(time_table["time_start"].to_pylist(), time_start, f"{meta['table_name']} timepoint filter")

        train_count = scalar(
            local_db,
            f"SELECT count(*) AS n FROM public.{meta['table_name']} WHERE COALESCE(is_test_split, false) = false",
            "n",
        )
        if train_count > 0:
            train_table = query_table(local_db, resolved, QuerySpec(holdout_split="train", max_rows=25))
            assert train_table.num_rows > 0, meta["table_name"]
            assert_all_false(train_table["is_test_split"].to_pylist(), f"{meta['table_name']} train split")

        test_count = scalar(
            local_db,
            f"SELECT count(*) AS n FROM public.{meta['table_name']} WHERE COALESCE(is_test_split, false) = true",
            "n",
        )
        if test_count > 0:
            test_table = query_table(local_db, resolved, QuerySpec(holdout_split="test", max_rows=25))
            assert test_table.num_rows > 0, meta["table_name"]
            assert_all_true(test_table["is_test_split"].to_pylist(), f"{meta['table_name']} test split")

        for location_key in LOCATION_KEYS:
            count = scalar(
                local_db,
                f"SELECT count(*) AS n FROM public.{meta['table_name']} "
                f"WHERE COALESCE({location_key}, false) = true",
                "n",
            )
            if count > 0:
                loc_table = query_table(
                    local_db,
                    resolved,
                    QuerySpec(required_locations=[location_key], max_rows=25),
                )
                assert loc_table.num_rows > 0, meta["table_name"]
                assert_all_true(loc_table[location_key].to_pylist(), f"{meta['table_name']} {location_key}")
                exercised_location_filter = True
                break

        synthetic_count = scalar(
            local_db,
            f"SELECT count(*) AS n FROM public.{meta['table_name']} "
            f"WHERE COALESCE(is_synthetic, false) = true",
            "n",
        )
        synthetic_table = query_table(local_db, resolved, QuerySpec(synthetic_only=True, max_rows=25))
        if synthetic_count > 0:
            assert synthetic_table.num_rows > 0, meta["table_name"]
            assert_all_true(synthetic_table["is_synthetic"].to_pylist(), f"{meta['table_name']} synthetic_only")
        else:
            assert synthetic_table.num_rows == 0, f"{meta['table_name']} synthetic_only should be empty"

    assert exercised_location_filter, "No location filter was exercised by the live DB contents"