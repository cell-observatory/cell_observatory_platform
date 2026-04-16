# import time
# import json
# import warnings
# from pathlib import Path
# from pprint import pprint

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pytest
# import ujson
# from hydra.utils import instantiate
# from omegaconf import OmegaConf
# from pyarrow import Table

# from cell_observatory_platform.tests.conftest import config

# warnings.filterwarnings("ignore")

# database_types = ["SupabaseDatabase"]  # List of database types to add to test matrix
# # database_types = ["SupabaseDatabase", "TrinoDatabase"]  # List of database types to add to test matrix


# def get_database_class(database_type):
#     if database_type == "TrinoDatabase":
#         return f"cell_observatory_platform.data.databases.trino_database.{database_type}"
#     elif database_type == "SupabaseDatabase":
#         return f"cell_observatory_platform.data.databases.supabase_database.{database_type}"
#     else:
#         raise ValueError(f"Invalid database type: {database_type}")


# @pytest.fixture(scope="module", params=database_types)
# def database(config, request):
#     database_type = request.param
#     config.experiment_name = f"test_{database_type}"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.input_shape = (16, 128, 128, 128, 2)
#     config.datasets.databases.dataset_layout_order = "TZYXC"
#     config.datasets.databases.use_cached_hypercubes_dataframe = False
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.outdir) / "database/hypercubes_dataframe.csv"
#     )
#     config.datasets.databases.fetch_hypercubes_dataframe = False
#     print(f"Initializing {config.datasets.databases._target_}...")
#     return instantiate(config.datasets.databases)


# def test_database_connection(database):
#     tables = database.list_tables()
#     assert tables is not None, "Connection to DB failed"
#     print(f"Available tables: {tables.values.squeeze()}")


# def test_all_database_tables(database):
#     tables = database.list_tables()
#     print(f"Available tables: {tables.values.squeeze()}")
#     assert len(tables) > 0, f"Zero tables were returned"


# def test_all_database_views(database):
#     views = database.list_views()
#     print(f"Available views: {views.values.squeeze()}")
#     assert len(views) > 0, f"Zero views were returned"


# @pytest.mark.parametrize(
#     "table_name",
#     [
#         "prepared",
#         "prepared_tiles",
#         "g_sheet_master_imaging_list",
#     ],
# )
# def test_table(database, table_name):
#     print(f"Testing table `{table_name}`...")
#     cols = database.get_columns(table_name)
#     num_cols = len(cols)
#     num_rows = database.count_rows(table_name)

#     assert num_cols > 1, f"Table `{table_name}` has {num_cols} column(s)"
#     assert num_rows > 0, f"Table `{table_name}` has {num_rows} row(s)"
#     print(f"Table `{table_name}` has {num_cols} column(s) and {num_rows} row(s).")
#     pprint(cols)


# def test_abc_data(database):
#     query = f""" SELECT id, output_folder, exists FROM prepared WHERE exists = TRUE """
#     table = database.execute_query(query)
#     num_rows, num_cols = table.shape
#     print(table)
#     print(f"Found {num_rows} rows.")
#     assert table.shape[0] > 0, "Zero hypercubes were returned"


# def test_prfs_data(database):
#     query = f""" SELECT id, output_folder, exists_prfs FROM prepared WHERE exists_prfs = TRUE """
#     table = database.execute_query(query)
#     num_rows, num_cols = table.shape
#     print(table)
#     print(f"Found {num_rows} rows.")
#     assert table.shape[0] > 0, "Zero hypercubes were returned"


# def test_aws_data(database):
#     query = f""" SELECT id, output_folder, exists_aws FROM prepared WHERE exists_aws = TRUE """
#     table = database.execute_query(query)
#     num_rows, num_cols = table.shape
#     print(table)
#     print(f"Found {num_rows} rows.")
#     assert table.shape[0] > 0, "Zero hypercubes were returned"


# def test_hypercubes_max_roi_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_rois=1)
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"

#     assert len(table["prepared_id"].unique()) == 1, "Only one ROI should be returned"


# def test_hypercubes_max_tiles_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_tiles=10)
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table.shape[0] > 0, f"Zero tiles were returned"
#     assert len(table["tile_name"].unique()) <= 10, "Only ten tiles should be returned"


# def test_hypercubes_max_hypercubes_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_hypercubes=100)
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"


# def test_hypercubes_list_roi_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_rois=1)
#     roi_list = table.prepared_id.unique().tolist()
#     print(f"test_hypercubes_list_roi_filter using {roi_list=}")
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, roi_list=roi_list)
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table["prepared_id"].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
#     assert table.shape[0] > 0, f"Zero ROIs were returned"


# def test_hypercubes_list_tiles_filter(database):
#     tile_list = ["000x_000y_000z.zarr", "000x_000y_001z.zarr", "000x_000y_002z.zarr"]

#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, tile_list=tile_list, max_hypercubes=1000)
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table["tile_name"].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"


# def test_hypercubes_list_filters(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_rois=1)
#     roi_list = table.prepared_id.unique().tolist()
#     print(f"test_hypercubes_list_roi_filter using {roi_list=}")
#     tile_list = ["000x_000y_000z.zarr", "000x_000y_001z.zarr", "000x_000y_002z.zarr"]

#     table = database.get_t_128_128_128_2_hypercubes(
#         num_timepoints=16,
#         roi_list=roi_list,
#         tile_list=["000x_000y_000z.zarr", "000x_000y_001z.zarr", "000x_000y_002z.zarr"],
#     )
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table["prepared_id"].isin(roi_list).all(), f"Only ROIs in {roi_list} should be returned"
#     assert table["tile_name"].isin(tile_list).all(), f"Only tiles in {tile_list} should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"


# def test_hypercubes_hpf_filter(database):
#     hpf_list = [72]
#     table = database.get_t_128_128_128_2_hypercubes(hpf_list=hpf_list, num_timepoints=16, max_hypercubes=100)
#     print(database.last_query)
#     print(table)

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert table["hpf"].isin(hpf_list).all(), f"Only hpf in {hpf_list} should be returned"


# def test_hypercubes_occ_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_hypercubes=100)
#     print(database.last_query)
#     table = database._aggregate(table)
#     table = database._apply_occupancy_threshold(
#         table, occupancy_threshold=0.9, occupancy_threshold_filter_type="min_ch0"
#     )

#     print(table)
#     print(table.columns)
#     print(table["min_occupancy_ratios_ch_0"])

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert (table["min_occupancy_ratios_ch_0"] >= 0.9).all(), f"Only occupancy_threshold >= 0.9 should be returned"


# def test_hypercubes_cdf_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(num_timepoints=16, max_hypercubes=100)
#     print(database.last_query)
#     table = database._aggregate(table)
#     table = database._apply_cdf_threshold(
#         table, cdf_threshold=150, cdf_target="90", cdf_threshold_filter_type="min_ch0"
#     )

#     print(table)
#     print(table.columns)
#     print(table["cdf_90_ch_0"])

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["time_size"] == 16).all(), "All time sizes should be 16"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert (table["cdf_90_ch_0"] >= 150).all(), f"Only cdf_90 >= 150 should be returned"


# @pytest.mark.skip("Supabase times out for this test.")
# def test_hypercubes_synthetic_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(synthetic_only=True, num_timepoints=1, max_hypercubes=100)
#     print(database.last_query)
#     print(table)

#     assert table["is_synthetic"].all(), "All hypercubes should be synthetic"
#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"


# @pytest.mark.skip("Supabase times out for this test.")
# def test_hypercubes_annotations_filter(database):
#     table = database.get_t_128_128_128_2_hypercubes(has_annotations=True, num_timepoints=1, max_hypercubes=100)
#     print(database.last_query)
#     print(table)

#     assert table["has_annotations"].all(), "All hypercubes should have annotations"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"

#     def _find_mask_bbox_dict(d):
#         if not isinstance(d, dict):
#             return {}
#         if "mask_bbox_dict" in d and isinstance(d["mask_bbox_dict"], dict) and bool(d["mask_bbox_dict"]):
#             return d["mask_bbox_dict"]
#         for v in d.values():
#             if isinstance(v, dict):
#                 found = _find_mask_bbox_dict(v)
#                 if found:
#                     return found
#         return {}

#     for i, pc_meta in enumerate(table["pc_metadata_json"]):
#         if isinstance(pc_meta, str):
#             try:
#                 pc_meta = json.loads(pc_meta)
#             except Exception as e:
#                 raise AssertionError(f"Failed to parse pc_metadata_json at row {i}: {e}")

#         assert isinstance(pc_meta, dict), f"pc_metadata_json at row {i} must be a dict"
#         mask = _find_mask_bbox_dict(pc_meta)
#         assert mask, f"mask_bbox_dict should not be empty at row {i}"

#     assert (table["channel_size"] == 2).all(), "All channel sizes should be 2"
#     assert (table["z_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["y_size"] == 128).all(), "All cube sizes should be 128"
#     assert (table["x_size"] == 128).all(), "All cube sizes should be 128"
#     assert table.shape[0] <= 100, "Only 100 hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"


# @pytest.mark.parametrize("max_hypercubes", [1000, 10000])
# @pytest.mark.parametrize("database_type", database_types)
# def test_1_128_128_128_2_hypercubes_database(config, database_type, max_hypercubes):
#     config.experiment_name = "test_1_128_128_128_2_hypercubes_database"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.input_shape = (128, 128, 128, 2)
#     config.datasets.databases.dataset_layout_order = "ZYXC"
#     num_timepoints = 1
#     config.datasets.databases.max_hypercubes = max_hypercubes
#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = False
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.outdir) / "database" / f"{config.experiment_name}.csv"
#     )

#     print(config.datasets.databases.hypercubes_dataframe_path)
#     print(f"Initializing {config.datasets.databases._target_}...")

#     start_time = time.time()
#     database = instantiate(config.datasets.databases)
#     table = database.hypercubes_dataframe
#     elapsed_time = time.time() - start_time
#     print(f"\nFetched {table.shape[0]} hypercubes in {elapsed_time:.3f} seconds.\n")
#     print(table)

#     assert (
#         table["time_size"] == num_timepoints
#     ).all(), f"All time sizes should be {num_timepoints}, found {table['time_size'].unique()}"

#     assert (
#         table.shape[0] <= config.datasets.databases.max_hypercubes
#     ), f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"


# @pytest.mark.parametrize("database_type", database_types)
# @pytest.mark.parametrize("max_hypercubes", [1000, 10000])
# def test_16_128_128_128_2_hypercubes_database(config, database_type, max_hypercubes):
#     config.experiment_name = "test_16_128_128_128_2_hypercubes_database"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.input_shape = (16, 128, 128, 128, 2)
#     num_timepoints = 16
#     config.datasets.databases.dataset_layout_order = "TZYXC"
#     config.datasets.databases.max_hypercubes = max_hypercubes
#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = False
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.outdir) / "database" / f"{config.experiment_name}.csv"
#     )

#     print(f"Initializing {config.datasets.databases._target_}...")

#     start_time = time.time()
#     database = instantiate(config.datasets.databases)
#     table = database.hypercubes_dataframe
#     elapsed_time = time.time() - start_time
#     print(f"\nFetched {table.shape[0]} hypercubes in {elapsed_time:.3f} seconds.\n")
#     print(table.columns)
#     print(table)

#     assert (
#         table["time_size"] == num_timepoints
#     ).all(), f"All time sizes should be {num_timepoints}, found {table['time_size'].unique()}"

#     assert (
#         table.shape[0] <= config.datasets.databases.max_hypercubes
#     ), f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"


# @pytest.mark.parametrize("database_type", database_types)
# def test_16_128_128_128_2_hypercubes_database_with_filters(config, database_type):
#     previous_config = config.datasets.databases.copy()
#     config.experiment_name = "test_16_128_128_128_2_hypercubes_database_with_filters"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.input_shape = (16, 128, 128, 128, 2)
#     num_timepoints = 16
#     config.datasets.databases.dataset_layout_order = "TZYXC"
#     config.datasets.databases.max_tiles = 2
#     config.datasets.databases.hpf_list = [72]
#     config.datasets.databases.max_hypercubes = 100
#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = False
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.outdir) / "database" / f"{config.experiment_name}.csv"
#     )

#     print(f"Initializing {config.datasets.databases._target_}...")
#     # pprint(OmegaConf.to_container(config, resolve=True))

#     database = instantiate(config.datasets.databases)
#     table = database.hypercubes_dataframe
#     print(table)

#     assert (table["time_size"] == num_timepoints).all(), f"All time sizes should be {num_timepoints}"
#     assert (
#         len(table["tile_name"].unique()) <= config.datasets.databases.max_tiles
#     ), f"Only {config.datasets.databases.max_tiles} tiles should be returned"
#     assert (
#         table.shape[0] <= config.datasets.databases.max_hypercubes
#     ), f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert (
#         table["hpf"].isin(config.datasets.databases.hpf_list).all()
#     ), f"Only hpf in {config.datasets.databases.hpf_list} should be returned"

#     config.datasets.databases = (
#         previous_config.copy()
#     )  #  Restore previous config state.  For the tests that follow, this will clear 'filters' we just added


# @pytest.mark.parametrize("database_type", database_types)
# @pytest.mark.parametrize(
#     "z_slices,y_slices,x_slices",
#     [
#         (128, 128, 128),
#         (128, 256, 256),
#         # (128, 384, 384),
#         # (128, 256, 512),
#         # (128, 384, 512),
#     ],
# )
# def test_aggregate_hypercubes(config, database_type, z_slices, y_slices, x_slices):
#     config.experiment_name = "test_aggregate_hypercubes"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.input_shape = (16, z_slices, y_slices, x_slices, 2)
#     num_timepoints = 16
#     config.datasets.databases.dataset_layout_order = "TZYXC"
#     config.datasets.databases.max_hypercubes = 10000
#     config.datasets.databases.max_rois = None
#     config.datasets.databases.max_tiles = None
#     config.datasets.databases.hpf_list = None
#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = False
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.outdir) / "database" / f"{config.experiment_name}.csv"
#     )

#     print(f"Initializing {config.datasets.databases._target_}...")
#     # pprint(OmegaConf.to_container(config, resolve=True))

#     database = instantiate(config.datasets.databases)
#     table = database.hypercubes_dataframe
#     print(table)

#     assert (table["time_size"] == num_timepoints).all(), f"All time sizes should be {num_timepoints}"
#     assert (
#         table.shape[0] <= config.datasets.databases.max_hypercubes
#     ), f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert table["first_pc_id"].unique().all(), f"`first_pc_id` should have unique values"
#     assert table["first_pc_id"].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"
#     assert (
#         table.shape[0] == config.datasets.databases.max_hypercubes
#     ), f"{config.datasets.databases.max_hypercubes} hypercubes should be returned"
#     assert (
#         table["occupancy_ratios_ch_0"].apply(len).unique()[0] == num_timepoints
#     ), "Should only have a single ratio for each timepoint"


# @pytest.mark.skip("test_csv_dataframe is only used for debugging.")
# @pytest.mark.parametrize("database_type", database_types)
# @pytest.mark.parametrize(
#     "z_slices,y_slices,x_slices",
#     [
#         (128, 128, 128),
#         (128, 256, 256),
#     ],
# )
# def test_csv_dataframe(config, database_type, z_slices, y_slices, x_slices):
#     config.experiment_name = "test_csv_dataframe"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.input_shape = (16, z_slices, y_slices, x_slices, 2)
#     num_timepoints = 16
#     config.datasets.databases.dataset_layout_order = "TZYXC"
#     config.datasets.databases.max_hypercubes = 100000
#     config.datasets.databases.max_rois = None
#     config.datasets.databases.max_tiles = None
#     config.datasets.databases.hpf_list = None
#     config.datasets.databases.has_annotations = False
#     config.datasets.databases.synthetic_only = False
#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = True
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.server_folder_path) / "databases" / "prepared_16_128_128_128_2_hypercube_view.csv"
#     )

#     print(f"Initializing {config.datasets.databases._target_}...")
#     # pprint(OmegaConf.to_container(config, resolve=True))

#     database = instantiate(config.datasets.databases)
#     table = database.hypercubes_dataframe
#     print(table)
#     # database.save_hypercubes_dataframe(hypercubes_dataframe_path=Path(config.paths.server_folder_path) / 'databases' / "prepared_16_128_128_128_2_hypercube_view.csv")

#     print(table.columns)

#     # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
#     # ax.hist(table["cdf_99_ch_0"].dropna(), bins=50, alpha=0.6, label="CDF-99")
#     # ax.hist(table["cdf_95_ch_0"].dropna(), bins=50, alpha=0.6, label="CDF-95")
#     # ax.hist(table["cdf_90_ch_0"].dropna(), bins=50, alpha=0.6, label="CDF-90")
#     # ax.hist(table["cdf_80_ch_0"].dropna(), bins=50, alpha=0.6, label="CDF-80")
#     # ax2.hist(table["min_occupancy_ratios_ch_0"].dropna(), bins=50, label="OCC")

#     # cdf_threshold, occ_threshold = 150, 0.9
#     # ax.axvline(x=cdf_threshold, color=f"r", linestyle="--", label="CDF threshold")
#     # for i, col in enumerate(["cdf_80_ch_0", "cdf_90_ch_0", "cdf_95_ch_0", "cdf_99_ch_0"]):
#     #     fraction_above_threshold = (table[col].dropna() >= cdf_threshold).sum() / table.shape[0]
#     #     ax.annotate(
#     #         f"{fraction_above_threshold:.2%} of {col} >= {cdf_threshold}",
#     #         xy=(cdf_threshold + 500, ax.get_ylim()[1] * 0.9 - i * ax.get_ylim()[1] * 0.25),
#     #         color=f"C{i}",
#     #         fontsize=10,
#     #         ha="left",
#     #     )

#     # ax2.axvline(x=occ_threshold, color=f"r", linestyle="--", label="OCC threshold")
#     # fraction_above_threshold = (table["min_occupancy_ratios_ch_0"].dropna() >= occ_threshold).sum() / table.shape[0]
#     # ax2.annotate(
#     #     f"{fraction_above_threshold:.2%} of min_occupancy_ratios_ch_0 >= {occ_threshold}",
#     #     xy=(occ_threshold - 0.05, ax.get_ylim()[1] * 0.9),
#     #     fontsize=10,
#     #     color="r",
#     #     ha="right",
#     # )

#     # ax.set_yscale("log")
#     # ax2.set_yscale("log")
#     # ax.set_xlabel("Camera counts")
#     # ax2.set_xlabel("OCC ratios")
#     # ax.set_ylabel("Frequency")
#     # ax2.set_ylabel("Frequency")

#     # plt.tight_layout()
#     # plt.savefig(f"cdf_histograms_t{num_timepoints}_z{z_slices}_y{y_slices}_x{x_slices}.png", dpi=300)

#     assert table.shape[0] > 0, f"Zero hypercubes were returned"
#     assert table["first_pc_id"].unique().all(), f"`first_pc_id` should have unique values"
#     assert table["first_pc_id"].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"
#     assert (
#         table["time_size"] == num_timepoints
#     ).all(), f"All time sizes should be {num_timepoints} found {table['time_size'].unique()}"
#     assert (
#         table["occupancy_ratios_ch_0"].apply(len).unique()[0] == num_timepoints
#     ), "Should only have a single ratio for each timepoint"


# @pytest.mark.skip("test_aggregate_hypercubes_metadata is only used for debugging.")
# @pytest.mark.parametrize("database_type", database_types)
# def test_aggregate_hypercubes_metadata(config, database_type):
#     config.experiment_name = "test_aggregate_hypercubes_metadata"
#     config.datasets.databases._target_ = get_database_class(database_type)

#     config.datasets.databases.input_shape = [128, 384, 1024, 2]
#     config.datasets.databases.dataset_layout_order = "ZYXC"

#     config.datasets.databases.max_hypercubes = None
#     config.datasets.databases.max_rois = None
#     config.datasets.databases.max_tiles = None
#     config.datasets.databases.hpf_list = None

#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = True
#     config.datasets.databases.with_hypercubes_dataframe = True
#     config.datasets.databases.has_annotations = True
#     config.datasets.databases.synthetic_only = True

#     db = instantiate(config.datasets.databases)
#     df = db.hypercubes_dataframe
#     assert not df.empty, "No hypercubes returned from DB."

#     important_cols = [
#         "prepared_id",
#         "tile_name",
#         "z_start",
#         "y_start",
#         "x_start",
#         "z_size",
#         "y_size",
#         "x_size",
#         "tile_z_end",
#         "tile_y_end",
#         "tile_x_end",
#         "tile_x_start",
#         "tile_y_start",
#         "tile_z_start",
#         "tile_time_size",
#         "tile_channel_size",
#     ]
#     important_cols = [c for c in important_cols if c in df.columns]
#     example_row = df[important_cols].head(1)

#     print(f"Aggregated Dataframe: {df}")
#     print(f"Columns: {df.columns.tolist()}")
#     print("\n=== Example Aggregated Row (first row) ===")
#     print(example_row.to_string(index=False))

#     bbox_cols = [c for c in df.columns if c.startswith("mask_bbox_dict_ch_")]
#     if not bbox_cols:
#         pytest.skip("No mask_bbox_dict_ch_* columns present in aggregated dataframe.")

#     col = None
#     for candidate in bbox_cols:
#         if df[candidate].notna().any():
#             col = candidate
#             break

#     if col is None:
#         pytest.skip("All mask_bbox_dict_ch_* columns are entirely null.")

#     df_nonnull = df[df[col].notna()]

#     volumes = []
#     frac_volumes = []
#     for _, row in df_nonnull.iterrows():
#         val = row[col]
#         if isinstance(val, str):
#             try:
#                 boxes_dict = ujson.loads(val)
#             except Exception:
#                 continue
#         elif isinstance(val, dict):
#             boxes_dict = val
#         else:
#             continue

#         if not boxes_dict:
#             continue

#         z_size = int(row["z_size"])
#         y_size = int(row["y_size"])
#         x_size = int(row["x_size"])
#         full_volume = z_size * y_size * x_size if (z_size and y_size and x_size) else None

#         for cell_id, box in boxes_dict.items():
#             if isinstance(box, dict):
#                 zmin = box.get("zmin")
#                 ymin = box.get("ymin")
#                 xmin = box.get("xmin")
#                 zmax = box.get("zmax")
#                 ymax = box.get("ymax")
#                 xmax = box.get("xmax")
#             elif isinstance(box, (list, tuple)) and len(box) == 6:
#                 zmin, ymin, xmin, zmax, ymax, xmax = box
#             else:
#                 continue

#             if None in (zmin, ymin, xmin, zmax, ymax, xmax):
#                 continue

#             dz = zmax - zmin
#             dy = ymax - ymin
#             dx = xmax - xmin
#             volume = dz * dy * dx

#             if volume <= 0:
#                 continue

#             volumes.append(volume)
#             if full_volume and full_volume > 0:
#                 frac_volumes.append(volume / full_volume)

#     assert len(volumes) > 0, f"Column {col} was non-null but no valid bbox entries were found after parsing."

#     volumes_arr = np.asarray(volumes, dtype=np.float64)
#     percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
#     vol_pct = np.percentile(volumes_arr, percentiles)

#     print("\n=== BBox volume distribution (voxels) ===")
#     print(f"num boxes: {len(volumes)}")
#     for p, v in zip(percentiles, vol_pct):
#         print(f"{p:3d}th percentile: {v:.2f}")

#     side_lengths = np.cbrt(volumes_arr)
#     side_pct = np.percentile(side_lengths, percentiles)

#     print("\n=== Approx. side-length distribution (cubic root of volume) ===")
#     for p, v in zip(percentiles, side_pct):
#         print(f"{p:3d}th percentile: {v:.2f} voxels")

#     if frac_volumes:
#         frac_arr = np.asarray(frac_volumes, dtype=np.float64)
#         frac_pct = np.percentile(frac_arr, percentiles)
#         print("\n=== BBox volume distribution (fraction of cube) ===")
#         for p, v in zip(percentiles, frac_pct):
#             print(f"{p:3d}th percentile: {v:.4f}")


# @pytest.mark.skip("test_tiles_dataframe is only used for debugging.")
# @pytest.mark.parametrize("database_type", database_types)
# def test_tiles_dataframe(config, database_type):
#     config.experiment_name = "test_tiles_dataframe"
#     config.datasets.databases._target_ = get_database_class(database_type)

#     config.datasets.databases.input_shape = [128, 384, 1024, 2]
#     config.datasets.databases.dataset_layout_order = "ZYXC"

#     config.datasets.databases.max_hypercubes = None
#     config.datasets.databases.max_rois = None
#     config.datasets.databases.max_tiles = None
#     config.datasets.databases.hpf_list = None

#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = True
#     config.datasets.databases.with_hypercubes_dataframe = True
#     config.datasets.databases.has_annotations = False
#     config.datasets.databases.synthetic_only = True

#     db = instantiate(config.datasets.databases)
#     df = db.hypercubes_dataframe

#     assert not df.empty, "No tiles returned from DB on tile path."

#     # assert (df["time_size"] == 1).all(), "Tile dataframe should have time_size == 1 after expansion."
#     assert (df["z_start"] == 0).all()
#     assert (df["y_start"] == 0).all()
#     assert (df["x_start"] == 0).all()

#     important_cols = [
#         "prepared_id",
#         "tile_name",
#         "z_start",
#         "y_start",
#         "x_start",
#         "z_size",
#         "y_size",
#         "x_size",
#         "tile_z_end",
#         "tile_y_end",
#         "tile_x_end",
#         "tile_x_start",
#         "tile_y_start",
#         "tile_z_start",
#         "tile_time_size",
#         "tile_channel_size",
#     ]
#     important_cols = [c for c in important_cols if c in df.columns]
#     example_row = df[important_cols].head(1)

#     print(f"Aggregated Dataframe: {df}")
#     print(f"Columns: {df.columns.tolist()}")
#     print("\n=== Example Aggregated Row (first row) ===")
#     print(example_row.to_string(index=False))

# @pytest.mark.parametrize("database_type", database_types)
# @pytest.mark.parametrize(
#     "t_timepoints,z_slices,y_slices,x_slices",
#     [
#         (1, 128, 128, 128),
#         (1, 128, 256, 256),
#         (16, 128, 128, 128),
#         (16, 128, 256, 256),
#     ],
# )
# def test_local_vs_prod_hypercubes_database(config, database_type, t_timepoints, z_slices, y_slices, x_slices):
#     config.experiment_name = f"test_local_vs_prod_{t_timepoints}d_{z_slices}z_{y_slices}y_{x_slices}x_hypercubes_database"
#     config.datasets.databases._target_ = get_database_class(database_type)
#     config.datasets.databases.max_hypercubes = 10000
#     config.datasets.databases.input_shape = (t_timepoints, z_slices, y_slices, x_slices, 2)
#     config.datasets.databases.dataset_layout_order = "TZYXC"
#     config.datasets.databases.fetch_hypercubes_dataframe = True
#     config.datasets.databases.use_cached_hypercubes_dataframe = False
#     config.datasets.databases.hypercubes_dataframe_path = (
#         Path(config.paths.outdir) / "database" / f"{config.experiment_name}.csv"
#     )

#     timings = {}
#     for dbname in ("local", "prod"):
#         config.datasets.databases.dbname = dbname
        
#         start_time = time.time()
#         database = instantiate(config.datasets.databases)
#         table = database.hypercubes_dataframe
#         timings[dbname] = time.time() - start_time
        
#         assert (table["time_size"] == t_timepoints).all(), f"All time sizes should be {t_timepoints}"
#         assert (
#             table.shape[0] <= config.datasets.databases.max_hypercubes
#         ), f"Only {config.datasets.databases.max_hypercubes} hypercubes should be returned"
#         assert table.shape[0] > 0, f"Zero hypercubes were returned"
#         assert table["first_pc_id"].unique().all(), f"`first_pc_id` should have unique values"
#         assert table["first_pc_id"].nunique() == table.shape[0], f"Each hypercube should have a unique `first_pc_id`"
#         assert (
#             table.shape[0] == config.datasets.databases.max_hypercubes
#         ), f"{config.datasets.databases.max_hypercubes} hypercubes should be returned"
    
#     print(f"{config.datasets.databases.max_hypercubes} x {config.datasets.databases.input_shape} HCs")    
#     print(f"prod: {timings['prod']:.3f}s, local: {timings['local']:.3f}s ({timings['prod'] / timings['local']:.2f}x)")
