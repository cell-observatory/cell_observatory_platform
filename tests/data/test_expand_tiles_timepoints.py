"""Unit tests for ParentDatabase._expand_tiles_timepoints_df."""

import pandas as pd
import pytest

from cell_observatory_platform.data.databases.base_database import ParentDatabase


def _db_num_timepoints_one():
    db = ParentDatabase.__new__(ParentDatabase)
    db.num_timepoints = 1
    return db


def test_expand_duplicates_rows_per_time_index():
    db = _db_num_timepoints_one()
    df = pd.DataFrame(
        {
            "tile_name": ["a.zarr", "b.zarr"],
            "time_start": [0, 0],
            "time_size": [3, 1],
            "z_size": [10, 10],
        }
    )
    out = ParentDatabase._expand_tiles_timepoints_df(db, df)
    assert len(out) == 4
    assert out["tile_name"].tolist() == ["a.zarr"] * 3 + ["b.zarr"]
    assert out["time_start"].tolist() == [0, 1, 2, 0]
    assert (out["time_size"] == 1).all()


def test_expand_respects_nonzero_time_start_base():
    db = _db_num_timepoints_one()
    df = pd.DataFrame(
        {
            "tile_name": ["t.zarr"],
            "time_start": [5],
            "time_size": [2],
        }
    )
    out = ParentDatabase._expand_tiles_timepoints_df(db, df)
    assert len(out) == 2
    assert out["time_start"].tolist() == [5, 6]
    assert (out["time_size"] == 1).all()


def test_all_time_size_one_returns_copy_unchanged_shape():
    db = _db_num_timepoints_one()
    df = pd.DataFrame({"time_size": [1], "time_start": [3], "tile_name": ["x.zarr"]})
    out = ParentDatabase._expand_tiles_timepoints_df(db, df)
    assert len(out) == 1
    assert int(out["time_start"].iloc[0]) == 3
    assert int(out["time_size"].iloc[0]) == 1


def test_adds_time_start_when_missing():
    db = _db_num_timepoints_one()
    df = pd.DataFrame({"tile_name": ["a.zarr"], "time_size": [2]})
    out = ParentDatabase._expand_tiles_timepoints_df(db, df)
    assert out["time_start"].tolist() == [0, 1]
    assert (out["time_size"] == 1).all()


def test_num_timepoints_gt_one_raises():
    db = ParentDatabase.__new__(ParentDatabase)
    db.num_timepoints = 2
    df = pd.DataFrame({"time_size": [2]})
    with pytest.raises(ValueError, match="num_timepoints"):
        ParentDatabase._expand_tiles_timepoints_df(db, df)
