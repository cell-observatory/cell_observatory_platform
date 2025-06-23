import pytest
from hydra.utils import instantiate
from hydra import initialize, compose
from pprint import pprint
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../../configs/data/databases"):
        cfg = compose(config_name="supabase_database")
    return cfg


@pytest.fixture(scope="module")
def database(cfg):
    return instantiate(cfg)


def test_database_connection(database):
    tables = database.list_tables()
    assert tables is not None, "Connection to DB failed"
    print(f"Available tables: {tables.values.squeeze()}")


def test_all_database_tables(database):
    tables = database.list_tables()
    for t in tables.values.squeeze():
        cols = database.count_columns(t)
        rows = database.count_rows(t)

        print(f"Table `{t}` has {cols} column(s) and {rows} row(s).")
        try:
            assert cols > 0
        except AssertionError:
            print(f"Table `{t}` has no columns. Check if the table exists in the database.")

        try:
            assert rows > 0
        except AssertionError:
            print(f"Table `{t}` is empty. Check access to this table in the database.")


def test_table(database, tablename='prepared'):
    print(f"Testing table `{tablename}`...")
    cols = database.get_columns(tablename)
    num_cols = len(cols)
    num_rows = database.count_rows(tablename)

    assert num_cols > 1, f"Table `{tablename}` has {num_cols} column(s)"
    assert num_rows > 0, f"Table `{tablename}` has {num_rows} row(s)"
    print(f"Table `{tablename}` has {num_cols} column(s) and {num_rows} row(s).")
    pprint(cols)

def test_prepared_tiles_table(database):
    test_table(database, tablename='prepared_tiles')

def test_prepared_cubes_table(database):
    test_table(database, tablename='prepared_cubes')

def test_prepared_tiles_view_table(database):
    test_table(database, tablename='prepared_tiles_view')

def test_get_32_128_128_128_2_hypercubes_with_10_rows(database):
    table = database.get_32_128_128_128_2_hypercubes(max_rows=10)
    print(database.last_query)
    print(table)
    assert table.shape[0] == 10, "Only ten rows should be returned"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )


def test_get_32_128_128_128_2_hypercubes_with_one_roi(database):
    table = database.get_32_128_128_128_2_hypercubes(max_rois=1)
    print(database.last_query)
    print(table)
    assert len(table['prepared_id'].unique()) == 1, "Only one ROI should be returned"
    assert len(table['tile_name'].unique()) > 1, "More than one tile should be returned"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_get_32_128_128_128_2_hypercubes_with_one_tile(database):
    table = database.get_32_128_128_128_2_hypercubes(max_tiles=1)
    print(database.last_query)
    print(table)

    assert len(table['prepared_id'].unique()) == 1, "Only one ROI should be returned"
    assert len(table['tile_name'].unique()) == 1, "Only one tile should be returned"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_get_32_128_128_128_2_hypercubes_with_ten_tile(database):
    table = database.get_32_128_128_128_2_hypercubes(max_tiles=10)
    print(database.last_query)
    print(table)

    assert len(table['prepared_id'].unique()) == 1, "Only one ROI should be returned"
    assert len(table['tile_name'].unique()) == 10, "Only ten tiles should be returned"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )
