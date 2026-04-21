"""Unit tests for local_metadata_store helpers (no live Postgres)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from cell_observatory_platform.data.databases.local_metadata_store import (
    LOCATION_SERVER_PATHS,
    MappedTable,
    QuerySpec,
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
