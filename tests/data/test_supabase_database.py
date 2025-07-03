import pytest
from hydra.utils import instantiate
from hydra import initialize, compose
from omegaconf import OmegaConf
from pprint import pprint
from pathlib import Path
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../configs/datasets/databases"):
        cfg = compose(config_name="supabase_database")
    return cfg


@pytest.fixture(scope="module")
def database(cfg):
    cfg.fetch_hypercubes_dataframe = False
    cfg.num_timepoints = 16
    cfg.hypercubes_dataframe_path = None
    pprint(OmegaConf.to_container(cfg, resolve=True))
    print(f"Initializing database...")
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

def test_create_2_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=2, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 2).all(), "All time sizes should be 2"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['occupancy_ratios_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_4_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=4, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 4).all(), "All time sizes should be 4"
    assert (table['cube_size'] == 128).all(), "All cube sizes should be 128"

    pd.testing.assert_series_equal(
        table['time_size'],
        table['occupancy_ratios_ch_0'].apply(len),
        check_dtype=False,
        check_names=False,
    )

def test_create_8_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=8, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 8).all(), "All time sizes should be 8"
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

def test_create_32_128_128_128_2_hypercubes(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=32, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] == 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 32).all(), "All time sizes should be 32"
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

def test_1_128_128_128_2_hypercubes_database(cfg):
    cfg.num_timepoints = 1
    cfg.max_hypercubes = 100
    cfg.fetch_hypercubes_dataframe = True
    cfg.hypercubes_dataframe_path = Path(__file__).parent.parent.parent.parent / 'databases' / 'test_1_128_128_128_2_hypercubes_database.csv'

    print(cfg.hypercubes_dataframe_path)
    print(f"Initializing database...")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    database = instantiate(cfg)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == cfg.num_timepoints).all(), f"All time sizes should be {cfg.num_timepoints}"
    assert table.shape[0] <= cfg.max_hypercubes, f"Only {cfg.max_hypercubes} hypercubes should be returned"

def test_16_128_128_128_2_hypercubes_database(cfg):
    cfg.num_timepoints = 16
    cfg.max_hypercubes = 100
    cfg.fetch_hypercubes_dataframe = True
    cfg.hypercubes_dataframe_path = Path(__file__).parent.parent.parent.parent / 'databases' / 'test_16_128_128_128_2_hypercubes_database.csv'

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    database = instantiate(cfg)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == cfg.num_timepoints).all(), f"All time sizes should be {cfg.num_timepoints}"
    assert table.shape[0] <= cfg.max_hypercubes, f"Only {cfg.max_hypercubes} hypercubes should be returned"

def test_16_128_128_128_2_hypercubes_database_with_filters(cfg):
    cfg.num_timepoints = 16
    cfg.max_rois = 2
    cfg.max_tiles = 2
    cfg.hpf_list = [72]
    cfg.max_hypercubes = 100
    cfg.fetch_hypercubes_dataframe = True
    cfg.hypercubes_dataframe_path = Path(__file__).parent.parent.parent.parent / 'databases' / 'test_16_128_128_128_2_hypercubes_database_with_filters.csv'

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    database = instantiate(cfg)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == cfg.num_timepoints).all(), f"All time sizes should be {cfg.num_timepoints}"
    assert len(table['prepared_id'].unique()) <= cfg.max_rois, f"Only {cfg.max_rois} ROI should be returned"
    assert len(table['tile_name'].unique()) <= cfg.max_tiles, f"Only {cfg.max_tiles} tiles should be returned"
    assert table.shape[0] <= cfg.max_hypercubes, f"Only {cfg.max_hypercubes} hypercubes should be returned"
    assert table['hpf'].isin(cfg.hpf_list).all(), f"Only hpf in {cfg.hpf_list} should be returned"

def test_16_128_128_128_2_hypercubes_database_10k(cfg):
    cfg.num_timepoints = 16
    cfg.max_hypercubes = 10000
    cfg.fetch_hypercubes_dataframe = True
    cfg.hypercubes_dataframe_path = Path(__file__).parent.parent.parent.parent / 'databases' / 'test_16_128_128_128_2_hypercubes_database_10k.csv'

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    database = instantiate(cfg)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == cfg.num_timepoints).all(), f"All time sizes should be {cfg.num_timepoints}"
    assert table.shape[0] <= cfg.max_hypercubes, f"Only {cfg.max_hypercubes} hypercubes should be returned"

def test_16_128_128_128_2_hypercubes_database_100k(cfg):
    cfg.num_timepoints = 16
    cfg.max_hypercubes = 100000
    cfg.fetch_hypercubes_dataframe = True
    cfg.hypercubes_dataframe_path = Path(__file__).parent.parent.parent.parent / 'databases' / 'test_16_128_128_128_2_hypercubes_database_100k.csv'

    print(f"Initializing database...")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    database = instantiate(cfg)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == cfg.num_timepoints).all(), f"All time sizes should be {cfg.num_timepoints}"
    assert table.shape[0] <= cfg.max_hypercubes, f"Only {cfg.max_hypercubes} hypercubes should be returned"
