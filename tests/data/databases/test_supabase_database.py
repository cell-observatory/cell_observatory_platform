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
    print(f"Available tables: {tables.values.squeeze()}")

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

def test_table(database, table_name='prepared'):
    print(f"Testing table `{table_name}`...")
    cols = database.get_columns(table_name)
    num_cols = len(cols)
    num_rows = database.count_rows(table_name)

    assert num_cols > 1, f"Table `{table_name}` has {num_cols} column(s)"
    assert num_rows > 0, f"Table `{table_name}` has {num_rows} row(s)"
    print(f"Table `{table_name}` has {num_cols} column(s) and {num_rows} row(s).")
    pprint(cols)

def test_g_sheet_master_imaging_list_table(database):
    test_table(database, table_name='g_sheet_master_imaging_list')

def test_prepared_table(database):
    test_table(database, table_name='prepared')

def test_prepared_tiles_table(database):
    test_table(database, table_name='prepared_tiles')

def test_prepared_cubes_table(database):
    test_table(database, table_name='prepared_cubes')

def test_get_32_128_128_128_2_hypercubes(database):
    table = database.get_32_128_128_128_2_hypercubes(max_hypercubes=10)
    print(database.last_query)
    print(table)
    assert table.shape[0] == 10, "Only ten rows should be returned"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_1_128_128_128_2_hypercubes(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=1, max_hypercubes=1000)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 1000, "Only ten rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 1).all(), "All time sizes should be 1"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_2_128_128_128_2_hypercubes(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=2, max_hypercubes=1000)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 1000, "Only ten rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 2).all(), "All time sizes should be 2"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_4_128_128_128_2_hypercubes(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=4, max_hypercubes=1000)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 1000, "Only ten rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 4).all(), "All time sizes should be 4"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_8_128_128_128_2_hypercubes(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=8, max_hypercubes=1000)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 1000, "Only ten rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 8).all(), "All time sizes should be 8"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_16_128_128_128_2_hypercubes(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=16, max_hypercubes=1000)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 1000, "Only ten rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_32_128_128_128_2_hypercubes(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=32, max_hypercubes=1000)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 1000, "Only ten rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 32).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['timepoints_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_hypercubes_max_roi_filter(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=16, max_rois=1)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert len(table['prepared_id'].unique()) == 1, "Only one ROI should be returned"
    assert len(table['tile_name'].unique()) > 1, "More than one tile should be returned"

def test_hypercubes_max_tiles_filter(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=16, max_tiles=10)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert len(table['tile_name'].unique()) == 10, "Only ten tiles should be returned"

def test_hypercubes_max_hypercubes_filter(database):
    table = database.create_multichannel_hypercube_table(num_timepoints=16, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table.shape[0] == 100, "Only 100 hypercubes should be returned"

def test_hypercubes_list_roi_filter(database):
    roi_list = [312]
    table = database.create_multichannel_hypercube_table(num_timepoints=16, roi_list=roi_list)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['prepared_id'].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
    assert len(table['tile_name'].unique()) > 1, "More than one tile should be returned"

def test_hypercubes_list_tiles_filter(database):
    tile_list = ['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']

    table = database.create_multichannel_hypercube_table(num_timepoints=16, tile_list=tile_list)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['tile_name'].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"

def test_hypercubes_list_filters(database):
    roi_list = [312]
    tile_list = ['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']

    table = database.create_multichannel_hypercube_table(
        num_timepoints=16,
        roi_list=[312],
        tile_list=['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']
    )
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 32"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['prepared_id'].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
    assert table['tile_name'].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"
