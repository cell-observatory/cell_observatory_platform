from pyarrow import Table
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pprint import pprint
from pathlib import Path
import pandas as pd

from cell_observatory_platform.tests.conftest import config

import warnings
warnings.filterwarnings("ignore")

database_types = ['SupabaseDatabase']  # List of database types to add to test matrix
# database_types = ['SupabaseDatabase', 'TrinoDatabase']  # List of database types to add to test matrix

def get_database_class(database_type):
    if database_type == 'TrinoDatabase':
        return f"data.databases.trino_database.{database_type}"
    elif database_type == 'SupabaseDatabase':
        return f"data.databases.supabase_database.{database_type}"
    else:
        raise ValueError(f"Invalid database type: {database_type}")

@pytest.fixture(scope="module", params=database_types)
def database(config, request):
    database_type = request.param
    config.experiment_name = f"test_{database_type}"
    config.datasets.databases._target_ = get_database_class(database_type) 
    config.datasets.databases.input_shape = (16, 128, 128, 128, 2)
    config.datasets.databases.dataset_layout_order = "TZYXC"
    config.datasets.databases.use_cached_hypercubes_dataframe = False
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / "database/hypercubes_dataframe.csv"
    config.datasets.databases.fetch_hypercubes_dataframe = False
    print(f"Initializing {config.datasets.databases._target_}...")
    return instantiate(config.datasets.databases)

def test_database_connection(database):
    tables = database.list_tables()
    assert tables is not None, "Connection to DB failed"
    print(f"Available tables: {tables.values.squeeze()}")

def test_all_database_tables(database):
    tables = database.list_tables()
    print(f"Available tables: {tables.values.squeeze()}")
    assert len(tables) > 0, f"Zero tables were returned"


def test_all_database_views(database):
    views = database.list_views()
    print(f"Available views: {views.values.squeeze()}")
    assert len(views) > 0, f"Zero views were returned"

@pytest.mark.parametrize(
    "table_name", 
    [
        'prepared', 
        'prepared_tiles', 
        'g_sheet_master_imaging_list',
    ]
)
def test_table(database, table_name):
    print(f"Testing table `{table_name}`...")
    cols = database.get_columns(table_name)
    num_cols = len(cols)
    num_rows = database.count_rows(table_name)

    assert num_cols > 1, f"Table `{table_name}` has {num_cols} column(s)"
    assert num_rows > 0, f"Table `{table_name}` has {num_rows} row(s)"
    print(f"Table `{table_name}` has {num_cols} column(s) and {num_rows} row(s).")
    pprint(cols)


def test_abc_data(database):
    query = f""" SELECT id, output_folder, exists FROM prepared WHERE exists = TRUE """ 
    table = database.execute_query(query)
    num_rows, num_cols = table.shape
    print(table)
    print(f"Found {num_rows} rows.")
    assert table.shape[0] > 0, "Zero hypercubes were returned"

def test_prfs_data(database):
    query = f""" SELECT id, output_folder, exists_prfs FROM prepared WHERE exists_prfs = TRUE """ 
    table = database.execute_query(query)
    num_rows, num_cols = table.shape
    print(table)
    print(f"Found {num_rows} rows.")
    assert table.shape[0] > 0, "Zero hypercubes were returned"

def test_aws_data(database):
    query = f""" SELECT id, output_folder, exists_aws FROM prepared WHERE exists_aws = TRUE """ 
    table = database.execute_query(query)
    num_rows, num_cols = table.shape
    print(table)
    print(f"Found {num_rows} rows.")
    assert table.shape[0] > 0, "Zero hypercubes were returned"


@pytest.mark.skip('Table is empty. Database connection not available')
@pytest.mark.parametrize("t", [1, 16])
def test_create_t_128_128_128_2_hypercubes(database, t):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=t, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert table.shape[0] <= 100, "Only 100 or less rows should be returned"

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == t).all(), f"All time sizes should be {t}"
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"

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
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"

    assert len(table['prepared_id'].unique()) == 1, "Only one ROI should be returned"

def test_hypercubes_max_tiles_filter(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_tiles=10)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table.shape[0] > 0, f"Zero tiles were returned"
    assert len(table['tile_name'].unique()) <= 10, "Only ten tiles should be returned"

def test_hypercubes_max_hypercubes_filter(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_hypercubes=100)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"
    assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"

def test_hypercubes_list_roi_filter(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_rois=1)
    roi_list = table.prepared_id.unique().tolist()
    print(f"test_hypercubes_list_roi_filter using {roi_list=}")
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, roi_list=roi_list)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table['prepared_id'].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
    assert table.shape[0] > 0, f"Zero ROIs were returned"

def test_hypercubes_list_tiles_filter(database):
    tile_list = ['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']

    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, tile_list=tile_list)
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table['tile_name'].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"

def test_hypercubes_list_filters(database):
    table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_rois=1)
    roi_list = table.prepared_id.unique().tolist()
    print(f"test_hypercubes_list_roi_filter using {roi_list=}")
    tile_list = ['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']

    table = database.get_t_128_128_128_2_hypercubes(
        num_timepoints=16,
        roi_list=roi_list,
        tile_list=['000x_000y_000z.zarr', '000x_000y_001z.zarr', '000x_000y_002z.zarr']
    )
    print(database.last_query)
    print(table)

    assert (table['channel_size'] == 2).all(), "All channel sizes should be 2"
    assert (table['time_size'] == 16).all(), "All time sizes should be 16"
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table['prepared_id'].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
    assert table['tile_name'].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"

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
    assert (table['z_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['y_size'] == 128).all(), "All cube sizes should be 128"
    assert (table['x_size'] == 128).all(), "All cube sizes should be 128"
    assert table['hpf'].isin(hpf_list).all(), f"Only hpf in {hpf_list} should be returned"
    assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"

@pytest.mark.skip('Table is empty. Database connection not available')
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

@pytest.mark.parametrize("database_type", database_types)
def test_1_128_128_128_2_hypercubes_database(config, database_type):
    config.experiment_name = "test_1_128_128_128_2_hypercubes_database"
    config.datasets.databases._target_ = get_database_class(database_type) 
    config.datasets.databases.input_shape = (128, 128, 128, 2)
    config.datasets.databases.dataset_layout_order = "ZYXC"
    num_timepoints = 1
    config.datasets.databases.max_hypercubes = 100
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.use_cached_hypercubes_dataframe = False
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(config.datasets.databases.hypercubes_dataframe_path)
    print(f"Initializing {config.datasets.databases._target_}...")
    pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == num_timepoints).all(), f"All time sizes should be {num_timepoints}"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"

@pytest.mark.parametrize("database_type", database_types)
@pytest.mark.parametrize("max_hypercubes", [100, 1000, 5000])
def test_16_128_128_128_2_hypercubes_database(config, database_type, max_hypercubes):
    config.experiment_name = "test_16_128_128_128_2_hypercubes_database"
    config.datasets.databases._target_ = get_database_class(database_type) 
    config.datasets.databases.input_shape = (16, 128, 128, 128, 2)
    num_timepoints = 16
    config.datasets.databases.dataset_layout_order = "TZYXC"    
    config.datasets.databases.max_hypercubes = max_hypercubes
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.use_cached_hypercubes_dataframe = False
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing {config.datasets.databases._target_}...")
    # pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == num_timepoints).all(), f"All time sizes should be {num_timepoints}"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"

@pytest.mark.parametrize("database_type", database_types)
def test_16_128_128_128_2_hypercubes_database_with_filters(config, database_type):
    previous_config = config.datasets.databases.copy()
    config.experiment_name = "test_16_128_128_128_2_hypercubes_database_with_filters"
    config.datasets.databases._target_ = get_database_class(database_type) 
    config.datasets.databases.input_shape = (16, 128, 128, 128, 2)
    num_timepoints = 16
    config.datasets.databases.dataset_layout_order = "TZYXC"
    config.datasets.databases.max_rois = 2
    config.datasets.databases.max_tiles = 2
    config.datasets.databases.hpf_list = [72]
    config.datasets.databases.max_hypercubes = 100
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.use_cached_hypercubes_dataframe = False
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing {config.datasets.databases._target_}...")
    # pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == num_timepoints).all(), f"All time sizes should be {num_timepoints}"
    assert len(table['prepared_id'].unique()) <= config.datasets.databases.max_rois, f"Only {config.datasets.databases.max_rois} ROI should be returned"
    assert len(table['tile_name'].unique()) <= config.datasets.databases.max_tiles, f"Only {config.datasets.databases.max_tiles} tiles should be returned"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"
    assert table['hpf'].isin(config.datasets.databases.hpf_list).all(), f"Only hpf in {config.datasets.databases.hpf_list} should be returned"

    config.datasets.databases = previous_config.copy()  #  Restore previous config state.  For the tests that follow, this will clear 'filters' we just added 

@pytest.mark.parametrize("database_type", database_types)
@pytest.mark.parametrize("z_slices,y_slices,x_slices", [
    (128, 128, 128),
    (128, 256, 256),
    # (128, 384, 384),
    # (128, 256, 512),
    # (128, 384, 512),
])
def test_aggregate_hypercubes(config, database_type, z_slices, y_slices, x_slices):
    config.experiment_name = "test_aggregate_hypercubes"
    config.datasets.databases._target_ = get_database_class(database_type) 
    config.datasets.databases.input_shape = (16, z_slices, y_slices, x_slices, 2)
    num_timepoints = 16
    config.datasets.databases.dataset_layout_order = "TZYXC"
    config.datasets.databases.max_hypercubes = 100
    config.datasets.databases.max_rois = None
    config.datasets.databases.max_tiles = None
    config.datasets.databases.hpf_list = None
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.use_cached_hypercubes_dataframe = False
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.outdir) / 'database' / f"{config.experiment_name}.csv"

    print(f"Initializing {config.datasets.databases._target_}...")
    # pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)

    assert (table['time_size'] == num_timepoints).all(), f"All time sizes should be {num_timepoints}"
    assert table.shape[0] <= config.datasets.databases.max_hypercubes, f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
    assert table.shape[0] > 0, f"Zero hypercubes were returned"
    assert table['first_pc_id'].unique().all(), f"`first_pc_id` should have unique values"
    assert table['first_pc_id'].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"
    assert table.shape[0] == config.datasets.databases.max_hypercubes, f"{config.datasets.databases.max_hypercubes} hypercubes should be returned"
    assert table['occupancy_ratios_ch_0'].apply(len).unique()[0] == num_timepoints, "Should only have a single ratio for each timepoint"
    
@pytest.mark.skip("Skipping test_csv_dataframe.")
@pytest.mark.parametrize("database_type", database_types)
@pytest.mark.parametrize("z_slices,y_slices,x_slices", [
    (128, 128, 128),
    (128, 256, 256),
    (128, 384, 384),
])
def test_csv_dataframe(config, database_type, z_slices, y_slices, x_slices):
    config.experiment_name = "test_csv_dataframe"
    config.datasets.databases._target_ = get_database_class(database_type)
    config.datasets.databases.input_shape = (16, z_slices, y_slices, x_slices, 2)
    num_timepoints = 16
    config.datasets.databases.dataset_layout_order = "TZYXC"    
    config.datasets.databases.max_hypercubes = 100000
    config.datasets.databases.max_rois = None
    config.datasets.databases.max_tiles = None
    config.datasets.databases.hpf_list = None
    config.datasets.databases.fetch_hypercubes_dataframe = True
    config.datasets.databases.use_cached_hypercubes_dataframe = True
    config.datasets.databases.hypercubes_dataframe_path = Path(config.paths.server_folder_path) / 'databases' / "prepared_16_128_128_128_2_hypercube_view.csv"

    print(f"Initializing {config.datasets.databases._target_}...")
    # pprint(OmegaConf.to_container(config, resolve=True))

    database = instantiate(config.datasets.databases)
    table = database.hypercubes_dataframe
    print(table)
    # database.save_hypercubes_dataframe(hypercubes_dataframe_path=Path(config.paths.server_folder_path) / 'databases' / "prepared_16_128_128_128_2_hypercube_view.csv")

    assert table.shape[0] > 0, f"Zero hypercubes were returned"
    assert table['first_pc_id'].unique().all(), f"`first_pc_id` should have unique values"
    assert table['first_pc_id'].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"
    assert (table['time_size'] == num_timepoints).all(), f"All time sizes should be {num_timepoints} found {table['time_size'].unique()}"
    assert table['occupancy_ratios_ch_0'].apply(len).unique()[0] == num_timepoints, "Should only have a single ratio for each timepoint"