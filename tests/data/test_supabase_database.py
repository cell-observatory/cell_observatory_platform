import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pprint import pprint
from pathlib import Path
import pandas as pd

from tests.conftest import config

import warnings
warnings.filterwarnings("ignore")

import pytest
pytest.skip("Database connection not available", allow_module_level=True)


@pytest.fixture(scope="module")
def database(config):
    config.experiment_name = "test_database"
    config.datasets.databases.num_timepoints = 16
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / "database/hypercubes_dataframe.csv"
    pprint(OmegaConf.to_container(config, resolve=True))

    print(f"Initializing database...")
    return instantiate(config.datasets.databases)


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

def test_create_1_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=1, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] <= 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 1).all(), "All time sizes should be 1"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['occupancy_ratios_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_16_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['occupancy_ratios_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_hypercubes_max_roi_filter(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_rois=1)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert len(table['prepared_id'].unique()) == 1, "Only one ROI should be returned"
    assert len(table['tile_name'].unique()) > 1, "More than one tile should be returned"

def test_hypercubes_max_tiles_filter(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_tiles=10)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert len(table['tile_name'].unique()) <= 10, "Only ten tiles should be returned"

def test_hypercubes_max_hypercubes_filter(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"

def test_hypercubes_list_roi_filter(database):
    roi_list = [312]
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, roi_list=roi_list)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['prepared_id'].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
    assert len(table['tile_name'].unique()) > 1, "More than one tile should be returned"

def test_hypercubes_list_tiles_filter(database):
    tile_list = ['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']

    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, tile_list=tile_list)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['tile_name'].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"

def test_hypercubes_list_filters(database):
    roi_list = [312]
    tile_list = ['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']

    table = database.get_t_128_128_128_2_hypercubes(
        num_timepoints=16,
        roi_list=[312],
        tile_list=['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']
    )
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['prepared_id'].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
    assert table['tile_name'].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"

def test_hypercubes_hpf_filter(database):
    hpf_list = [72]
    table = database.get_t_128_128_128_2_hypercubes(
        hpf_list=hpf_list,
        num_timepoints=16,
        max_hypercubes=100
    )
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    assert table['hpf'].isin(hpf_list).all(), f"Only hpf in {hpf_list} should be returned"
    assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"

def test_get_t_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(
        num_timepoints=1,
        max_rois=1,
        max_tiles=2,
        hpf_list=[72],
        max_hypercubes=10
    )
    print(database.last_query)
    print(table)

    assert table.shape[0] <= 10, "Only ten or less rows should be returned"
    assert len(table['prepared_id'].unique()) <= 1, "Only one ROI should be returned"
    assert len(table['tile_name'].unique()) <= 2, "More than one tile should be returned"
    assert table['hpf'].isin([72]).all(), f"Only hpf in {[72]} should be returned"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['occupancy_ratios_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_1_128_128_128_2_hypercubes_database(config):
    config.experiment_name = "test_1_128_128_128_2_hypercubes_database"
    config.datasets.databases.num_timepoints = 1
    config.datasets.databases.max_hypercubes = 100
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(config.datasets.databases.hypercubes_dataframe_path)
    print(f"Initializing database...")
    pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == config.datasets.databases.num_timepoints).all(), f"All time sizes should be {config.datasets.databases.num_timepoints}"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"

def test_16_128_128_128_2_hypercubes_database(config):
    config.experiment_name = "test_16_128_128_128_2_hypercubes_database"
    config.datasets.databases.num_timepoints = 16
    config.datasets.databases.max_hypercubes = 100
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == config.datasets.databases.num_timepoints).all(), f"All time sizes should be {config.datasets.databases.num_timepoints}"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"

def test_16_128_128_128_2_hypercubes_database_with_filters(config):
    config.experiment_name = "test_16_128_128_128_2_hypercubes_database_with_filters"
    config.datasets.databases.num_timepoints = 16
    config.datasets.databases.max_rois = 2
    config.datasets.databases.max_tiles = 2
    config.datasets.databases.hpf_list = [72]
    config.datasets.databases.max_hypercubes = 100
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == config.datasets.databases.num_timepoints).all(), f"All time sizes should be {config.datasets.databases.num_timepoints}"
    assert len(table['prepared_id'].unique()) <= config.datasets.databases.max_rois, f"Only {config.datasets.databases.max_rois} ROI should be returned"
    assert len(table['tile_name'].unique()) <= config.datasets.databases.max_tiles, f"Only {config.datasets.databases.max_tiles} tiles should be returned"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
    assert table['hpf'].isin(config.datasets.databases.hpf_list).all(), f"Only hpf in {config.datasets.databases.hpf_list} should be returned"

def test_16_128_128_128_2_hypercubes_database_10k(config):
    config.experiment_name = "test_16_128_128_128_2_hypercubes_database_10k"
    config.datasets.databases.num_timepoints = 16
    config.datasets.databases.max_hypercubes = 10000
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == config.datasets.databases.num_timepoints).all(), f"All time sizes should be {config.datasets.databases.num_timepoints}"

    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"

    assert table['first_pc_id'].unique().all(), f"`first_pc_id` should have unique value"

    assert table['first_pc_id'].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"


def test_16_128_128_128_2_hypercubes_database_100k(config):
    config.experiment_name = "test_16_128_128_128_2_hypercubes_database_100k"
    config.datasets.databases.num_timepoints = 16
    config.datasets.databases.max_hypercubes = 100000
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == config.datasets.databases.num_timepoints).all(), f"All time sizes should be {config.datasets.databases.num_timepoints}"

    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"

    assert table['first_pc_id'].unique().all(), f"`first_pc_id` should have unique values"

    assert table['first_pc_id'].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"
