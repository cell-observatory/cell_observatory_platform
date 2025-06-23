import sys
import logging

import pandas as pd


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseDatabase:
    def __init__(self):
        # dummy dataset while we are working on 
        # getting the SupaBase DB online
        self.data_table = pd.DataFrame(
            data = dict(
            server_folder=["/clusterfs/vast/Data/cell_observatory_training_datasets/", 
                           "/clusterfs/vast/Data/cell_observatory_training_datasets/",
                           "/clusterfs/vast/Data/cell_observatory_training_datasets/",
                           "/clusterfs/vast/Data/cell_observatory_training_datasets/",
                           "/clusterfs/vast/Data/cell_observatory_training_datasets/",
                           "/clusterfs/vast/Data/cell_observatory_training_datasets/"],
            output_folder=["2025/6/9/20250324_mem_histone/fish1_72hpf/roi1", 
                           "2025/6/9/20250324_mem_histone/fish1_72hpf/roi1",
                           "2025/6/9/20250324_mem_histone/fish1_72hpf/roi1",
                           "2025/6/9/20250324_mem_histone/fish1_72hpf/roi1",
                           "2025/6/9/20250324_mem_histone/fish1_72hpf/roi1",
                           "2025/6/9/20250324_mem_histone/fish1_72hpf/roi1"],
            tile_name=["000x_000y_000z.zarr", 
                       "000x_000y_000z.zarr",
                       "000x_000y_000z.zarr",
                       "000x_000y_000z.zarr",
                       "000x_000y_000z.zarr",
                       "000x_000y_000z.zarr"],
            time_start=[0, 32, 64, 0, 32, 64],
            z_start=[0, 0, 0, 0, 0, 0],
            y_start=[0, 0, 0, 0, 0, 0],
            x_start=[0, 0, 0, 128, 128, 128],
            cube_size=[128, 128, 128, 128, 128, 128],
            channel_size=[2, 2, 2, 2, 2, 2],
            time_size=[32, 32, 32, 32, 32, 32]
            )
        )