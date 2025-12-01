import sys
import time
import logging
from pathlib import Path
from warnings import filters
from abc import abstractmethod
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

import ujson
import connectorx as cx
from sqlite3 import NotSupportedError

import numpy as np
import pandas as pd
import polars as pl

from cell_observatory_platform.data.io import (
    load_hypercubes_dataframe,
    load_tiles_dataframe,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ParentDatabase:
    def __init__(
        self,
        input_shape: tuple,
        dataset_layout_order: str,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Sequence[int]] = None,
        roi_list: Optional[Sequence[int]] = None,
        tile_list: Optional[Sequence[str]] = None,
        timepoint_list: Optional[Iterable[int]] = None,
        dbname: Literal["staging", "prod"] = "prod",
        dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
        verbose: bool = False,
        fetch_hypercubes_dataframe: bool = True,
        hypercubes_dataframe_path: Optional[Path] = None,
        use_cached_hypercubes_dataframe: Optional[bool] = False,
        protocol: cx.Protocol | None = None,  # Literal["csv", "binary", "cursor", "simple", "text"]
        max_partitions: Optional[int] = 10,
        server_folder_path: Optional[Path | str] = None,
        occupancy_threshold: Optional[float] = None,
        occupancy_threshold_filter_type: str = "min_all",
        base_cube_size: Optional[int] = 128,
        valid_z_sizes: Optional[Sequence[int]] = [128],
        valid_y_sizes: Optional[Sequence[int]] = [128, 256, 384],
        valid_x_sizes: Optional[Sequence[int]] = [128, 256, 384, 512, 640, 896, 1024],
        synthetic_only: bool = False,
        has_annotations: bool = False,
        with_hypercubes_dataframe: bool = True,
        mask_channel: Optional[int] = None,
    ):
        """
        A class for accessing database and retrieving hypercubes.
        It'll check database for existing views or create them if they don't exist based on the given `num_timepoints`.
        The results are stored in a pandas dataframe `self.hypercubes_dataframe`
        unless `fetch_hypercubes_dataframe` is set to False.

        Args:
            max_rois: maximum number of ROIs (each ROI can have dozens of tiles)
            max_tiles: maximum number of tiles (each tile can have thousands of hypercubes)
            max_hypercubes: maximum number of hypercubes to return
            hpf_list: list of specific HPFs (hours-post-fertilization in hours) to filter
            roi_list: list of specific ROIs to filter
            tile_list: list of specific tiles to filter
            timepoint_list: list of specific timepoints to filter
            dbname: database name ('staging' or 'prod')
            dotenv_path: path to .env file with URIs to access Supabase
            verbose: whether to print debug messages
            fetch_hypercubes_dataframe: this will automatically initialize the database based on the provided parameters
                (only turn off for debugging or if the database is already initialized)
            hypercubes_dataframe_path: path to save hypercubes dataframe
            use_cached_hypercubes_dataframe: if True, will use the cached hypercubes dataframe from the given path
            protocol: The protocol used to fetch data from source (default is 'binary', can also be 'csv' or 'cursor')
            max_partitions: The maximum number of threads to fetch queries at once
            server_folder_path: path to override default server folder found in the supabase database
                update this path based on where the data is stored on your local machine
            occupancy_threshold: to filter our hypercubes with less than this occupancy ratio (0.0-1.0)
            synthetic_only: a toggle to only query synthetic hypercubes
            has_annotations: a toggle to only query hypercubes with annotations
            with_hypercubes_dataframe: whether to use hypercubes dataframe or tiles dataframe
            mask_channel: index of the mask channel in the data (if any)
        """
        self.verbose = verbose

        if hypercubes_dataframe_path is None:
            self.hypercubes_dataframe_path = (
                Path(__file__).parent.parent.parent / "databases" / "default_hypercubes_dataframe.csv"
            )
        else:
            self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)

        self.input_shape = input_shape

        self.dbname = dbname
        self.dotenv_path = dotenv_path

        self.max_rois = max_rois
        self.max_tiles = max_tiles
        
        self.hpf_list = hpf_list
        self.roi_list = roi_list
        self.tile_list = tile_list
        
        self.timepoint_list = timepoint_list
        
        self.fetch_hypercubes_dataframe = fetch_hypercubes_dataframe
        self.use_cached_hypercubes_dataframe = use_cached_hypercubes_dataframe
        
        self.protocol = protocol
        self.max_partitions = max_partitions
        
        self.server_folder_path = server_folder_path
        
        self.occupancy_threshold = occupancy_threshold
        self.occupancy_threshold_filter_type = occupancy_threshold_filter_type
        
        self.mask_channel = mask_channel
        self.synthetic_only = synthetic_only
        self.has_annotations = has_annotations
        self.dataset_layout_order = dataset_layout_order

        self.with_hypercubes_dataframe = with_hypercubes_dataframe
    
        self.base_cube_size = base_cube_size
        self.num_timepoints, z_slices, y_slices, x_slices = self._get_slices_from_layout_order(
            input_format=self.dataset_layout_order, input_shape=self.input_shape
        )

        if self.with_hypercubes_dataframe:
            if z_slices not in valid_z_sizes:
                raise NotSupportedError(f"{z_slices=} is not supported yet, please chose from {valid_z_sizes}")
            else:
                self.z_slices = z_slices

            if y_slices not in valid_y_sizes:
                raise NotSupportedError(f"{y_slices=} is not supported yet, please chose from {valid_y_sizes}")
            else:
                self.y_slices = y_slices

            if x_slices not in valid_x_sizes:
                raise NotSupportedError(f"{x_slices=} is not supported yet, please chose from {valid_x_sizes}")
            else:
                self.x_slices = x_slices

            if self.z_slices != self.base_cube_size or self.y_slices != self.base_cube_size \
                or self.x_slices != self.base_cube_size:
                self.max_hypercubes = max_hypercubes
                if max_hypercubes is None:
                    self.max_hypercubes_128 = None
                else:
                    self.max_hypercubes_128 = (
                        max_hypercubes
                        * (self.z_slices // self.base_cube_size)
                        * (self.y_slices // self.base_cube_size)
                        * (self.x_slices // self.base_cube_size)
                    )
                    print(
                        f"Requesting {self.max_hypercubes_128 - max_hypercubes} extra hypercubes \
                            to get {max_hypercubes} hypercubes after aggregation"
                    )
            else:
                self.max_hypercubes = max_hypercubes
                self.max_hypercubes_128 = max_hypercubes
        
        else:
            self.max_hypercubes = max_hypercubes
            self.max_hypercubes_128 = max_hypercubes

        self._database_url = self._load_uri()

        if self.fetch_hypercubes_dataframe:
            self.hypercubes_dataframe = self._fetch_samples_dataframe()
        else:
            self.hypercubes_dataframe = None

    def _fetch_samples_dataframe(self) -> pd.DataFrame:
        if self.with_hypercubes_dataframe:
            self.hypercubes_dataframe = self._fetch_hypercubes_dataframe()
        else:
            self.hypercubes_dataframe = self._fetch_tiles_dataframe()
        return self.hypercubes_dataframe
    
    def _fetch_hypercubes_dataframe(self) -> pd.DataFrame:
        if self.use_cached_hypercubes_dataframe:
            # we assume that hypercubes_dataframe_path has a valid csv
            self.hypercubes_dataframe, self.hypercubes_dataframe_config = load_hypercubes_dataframe(
                hypercubes_dataframe_path=self.hypercubes_dataframe_path,
                server_folder_path=self.server_folder_path,
                max_rois=self.max_rois,
                max_tiles=self.max_tiles,
                max_hypercubes=self.max_hypercubes_128,
                hpf_list=self.hpf_list,
                roi_list=self.roi_list,
                tile_list=self.tile_list,
                timepoint_list=self.timepoint_list,
                occupancy_threshold=self.occupancy_threshold,
                occupancy_threshold_filter_type=self.occupancy_threshold_filter_type,
                synthetic_only=self.synthetic_only,
                has_annotations=self.has_annotations,
            )

        else:
            self.hypercubes_dataframe = self.get_t_128_128_128_2_hypercubes(
                num_timepoints=self.num_timepoints,
                max_rois=self.max_rois,
                max_tiles=self.max_tiles,
                max_hypercubes=self.max_hypercubes_128,
                hpf_list=self.hpf_list,
                roi_list=self.roi_list,
                tile_list=self.tile_list,
                timepoint_list=self.timepoint_list,
                occupancy_threshold=self.occupancy_threshold,
                synthetic_only=self.synthetic_only,
                has_annotations=self.has_annotations,
            )

            self.save_hypercubes_dataframe(hypercubes_dataframe_path=self.hypercubes_dataframe_path)

        if self.server_folder_path is not None:
                self.hypercubes_dataframe["server_folder"] = self.server_folder_path

        if self.z_slices != self.base_cube_size or self.y_slices != self.base_cube_size or self.x_slices != self.base_cube_size:
            print(f"Size of volume axes not equal to base cube size of {self.base_cube_size}, aggregating hypercubes...")
        else:
            print(f"Volume axes equal base cube size {self.base_cube_size}, running metadata aggregation only...")

        self.aggregate_hypercubes(
            z_slices=self.z_slices,
            y_slices=self.y_slices,
            x_slices=self.x_slices,
        )

        if any(self.hypercubes_dataframe["time_size"] != self.num_timepoints):
            print(
                f"`time_sizes` for all rows in the dataframe should be {self.num_timepoints} \
                    found {self.hypercubes_dataframe['time_size'].unique()}"
            )
            print("Overriding values in the dataframe")
            self.hypercubes_dataframe["time_size"] = self.num_timepoints

        # NOTE: may be reset below in check_hypercube_sizes
        self.hypercubes_dataframe["z_size"] = self.z_slices
        self.hypercubes_dataframe["y_size"] = self.y_slices
        self.hypercubes_dataframe["x_size"] = self.x_slices

        print(f"Loading ROIs dataframe to check hypercube sizes...")
        # FIXME: assumes that all tiles per ROI share the same shape
        #        which is true currently but unsafe, we should adjust logic
        #        to get tile shapes per tile
        self.rois_dataframe = self.get_rois_dataframe()
        print(f"Checking hypercube sizes against ROIs dataframe for {len(self.hypercubes_dataframe)} hypercubes...")
        self.hypercubes_dataframe = self.check_hypercube_sizes(
            df=self.hypercubes_dataframe,
            shape_df=self.rois_dataframe,
            layout=self.dataset_layout_order,
        )
        self.hypercubes_dataframe = self.hypercubes_dataframe.head(self.max_hypercubes)
        print(f"Final length of hypercubes dataframe: {len(self.hypercubes_dataframe)}")

        return self.hypercubes_dataframe

    def _fetch_tiles_dataframe(self) -> pd.DataFrame:
        """
        Fetch a dataframe where each row is a full tile (not a hypercube), either
        from a cached CSV or directly from the database.
        """
        if self.use_cached_hypercubes_dataframe:
            (
                self.hypercubes_dataframe,
                self.hypercubes_dataframe_config,
            ) = load_tiles_dataframe(
                hypercubes_dataframe_path=self.hypercubes_dataframe_path,
                server_folder_path=self.server_folder_path,
                max_rois=self.max_rois,
                max_tiles=self.max_tiles,
                hpf_list=self.hpf_list,
                roi_list=self.roi_list,
                tile_list=self.tile_list,
                timepoint_list=self.timepoint_list,
                synthetic_only=self.synthetic_only,
                has_annotations=self.has_annotations,
            )
        else:
            self.hypercubes_dataframe = self.get_tiles(
                max_rois=self.max_rois,
                max_tiles=self.max_tiles,
                hpf_list=self.hpf_list,
                roi_list=self.roi_list,
                tile_list=self.tile_list,
                timepoint_list=self.timepoint_list,
                synthetic_only=self.synthetic_only,
                has_annotations=self.has_annotations,
            )
            self.save_hypercubes_dataframe(hypercubes_dataframe_path=self.hypercubes_dataframe_path)

        if self.server_folder_path is not None:
                self.hypercubes_dataframe["server_folder"] = self.server_folder_path

        # handle time granularity: expand time_size into per-timepoint rows
        self.hypercubes_dataframe = self._expand_tiles_timepoints_df(self.hypercubes_dataframe)

        if self.timepoint_list is not None and "time_start" in self.hypercubes_dataframe.columns:
            self.hypercubes_dataframe = self.hypercubes_dataframe[
                self.hypercubes_dataframe["time_start"].isin(self.timepoint_list)
            ].reset_index(drop=True)

        print(f"Final length of tiles dataframe: {len(self.hypercubes_dataframe)}")
        return self.hypercubes_dataframe

    def get_tiles(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        hpf_list: Optional[Sequence[int]] = None,
        roi_list: Optional[Sequence[int]] = None,
        tile_list: Optional[Sequence[str]] = None,
        timepoint_list: Optional[Iterable[int]] = None,
        synthetic_only: bool = False,
        has_annotations: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch one row per (prepared_id, tile_name) from the DB and add
        with ROI-level metadata.
        """
        filters = self._filters_to_string(
            table_name="prepared_tiles_view",
            table_name_shortcut="ptv",
            max_rois=max_rois,
            max_tiles=max_tiles,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list,
            timepoint_list=None,  # timepoints handled after expansion
            synthetic_only=synthetic_only,
            has_annotations=has_annotations,
        )

        base_cols = [
            "prepared_id",
            "tile_name",
            "server_folder",
            "output_folder",
            "hpf",
            # NOTE: not yet supported in database
            # "pc_metadata_json",
            "exists",
            "exists_prfs",
            "exists_aws",
            "is_synthetic",
        ]

        query = f"""
            SELECT
                {', '.join(f'ptv.{c}' for c in base_cols)}
            FROM prepared_tiles_view ptv
            {filters}
        """

        table = self.execute_query(query)

        # pull ROI-level metadata
        rois_df = self.get_rois_dataframe()
        table = table.merge(
            rois_df[
                [
                    "prepared_id",
                    "tile_z_start",
                    "tile_y_start",
                    "tile_x_start",
                    "tile_z_end",
                    "tile_y_end",
                    "tile_x_end",
                    "tile_time_size",
                    "tile_channel_size",
                ]
            ],
            on="prepared_id",
            how="left",
        )

        table["z_start"], table["y_start"], table["x_start"], table["time_start"] = 0, 0, 0, 0

        table["z_size"] = table["tile_z_end"] - table["tile_z_start"]
        table["y_size"] = table["tile_y_end"] - table["tile_y_start"]
        table["x_size"] = table["tile_x_end"] - table["tile_x_start"]

        # HOTFIX: synthetic tiles have incorrect tile_*_end bounds in the DB.
        # For synthetic data, the true tile size should be
        #   tile_*_end - tile_*_start + base_cube_size (e.g. +128).
        # --- --- --- ---
        # if synthetic_only or self.synthetic_only:
        #     table["z_size"] = table["z_size"] + self.base_cube_size
        #     table["y_size"] = table["y_size"] + self.base_cube_size
        #     table["x_size"] = table["x_size"] + self.base_cube_size
        # --- --- --- ---

        table["time_size"] = table["tile_time_size"]
        table["channel_size"] = table["tile_channel_size"]

        num_rows, num_cols = table.shape
        print(f"\nRetrieved {num_rows} tiles. \t Retrieved {num_cols} columns.")
        
        return table

    def _expand_tiles_timepoints_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For tile-based training, if num_timepoints == 1 but the tiles in the
        dataframe have time_size > 1, duplicate each tile row once per
        timepoint and:
          - set time_start = base_time_start + t
          - set time_size = 1
        TODO: support num_timepoints > 1 case
        """
        if "time_size" not in df.columns:
            raise ValueError("Dataframe must have a time_size column to expand timepoints.")
        if self.num_timepoints != 1:
            raise ValueError("Currently tile based training only supports num_timepoints=1.")

        if all(df["time_size"] == 1):
            return df
        else:
            raise NotImplementedError("Expanding tiles for time_size > 1 is not implemented yet.")

    def _get_slices_from_layout_order(self, input_format: str, input_shape: tuple):
        if input_format == "TZYXC":
            num_timepoints = input_shape[0]
            z_slices = input_shape[1]
            y_slices = input_shape[2]
            x_slices = input_shape[3]
        elif input_format == "ZYXC":
            num_timepoints = 1
            z_slices = input_shape[0]
            y_slices = input_shape[1]
            x_slices = input_shape[2]
        else:
            raise NotSupportedError(f"Input format {input_format} is not supported yet.")
        return num_timepoints, z_slices, y_slices, x_slices

    def get_rois_dataframe(self) -> pd.DataFrame:
        roi_csv = self.hypercubes_dataframe_path.with_name(f"{self.hypercubes_dataframe_path.stem}_rois.csv")
        if (not self.use_cached_hypercubes_dataframe) or (not roi_csv.exists()):
            query = f"""
                SELECT id,
                    x_start, y_start, z_start,
                    z_end, y_end, x_end,
                    time_size, channel_size, is_synthetic
                FROM prepared
            """
            rois_df = self.execute_query(query)
            rois_df = rois_df.rename(
                columns={
                    "id": "prepared_id",
                    "z_end": "tile_z_end",
                    "y_end": "tile_y_end",
                    "x_end": "tile_x_end",
                    "z_start": "tile_z_start",
                    "y_start": "tile_y_start",
                    "x_start": "tile_x_start",
                    "time_size": "tile_time_size",
                    "channel_size": "tile_channel_size",
                }
            )

            # HOTFIX: synthetic ROIs have incorrect tile_*_end bounds in the DB.
            # For synthetic data, the true tile size should be
            #   tile_*_end - tile_*_start + base_cube_size (e.g. +128).
            # --- --- --- ---
            if "is_synthetic" in rois_df.columns:
                mask = rois_df["is_synthetic"] == True
                rois_df.loc[mask, "tile_z_end"] = rois_df.loc[mask, "tile_z_end"] + self.base_cube_size
                rois_df.loc[mask, "tile_y_end"] = rois_df.loc[mask, "tile_y_end"] + self.base_cube_size
                rois_df.loc[mask, "tile_x_end"] = rois_df.loc[mask, "tile_x_end"] + self.base_cube_size
            # --- --- --- ---

            rois_df.to_csv(roi_csv, index=True, header=True)
            print(f"Saved roi dataframe to {roi_csv}")

        else:
            rois_df = pd.read_csv(roi_csv, index_col=0)

        return rois_df

    @abstractmethod
    def _load_uri(self) -> str:
        """To override"""
        pass

    def _choose_filter(
        self,
        rois: Optional[Sequence[int | str]] = None,
        tiles: Optional[Sequence[str]] = None,
        timepoints: Optional[Iterable[int]] = None,
        table_name: str = "ptv",
        idx_col: str = "prepared_id",
    ) -> str:

        def _sql_in_list(values):
            out = []
            for v in values:
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.isnumeric()):
                    out.append(str(v))
                else:
                    out.append("'" + str(v).replace("'", "''") + "'")
            return "(" + ",".join(out) + ")"

        assert rois is not None or tiles is not None, "At least one of rois or tiles must be provided"

        clauses = []
        if rois is not None:
            rois_list = list(rois)
            clauses.append(f"{table_name}.{idx_col} IN {_sql_in_list(rois_list)}")

        if tiles is not None:
            tiles_list = list(tiles)
            clauses.append(f"{table_name}.tile_name IN {_sql_in_list(tiles_list)}")

        if timepoints is not None:
            timepoints_list = list(timepoints)
            clauses.append(f"{table_name}.time_start IN {_sql_in_list(timepoints_list)}")

        return "WHERE " + " AND ".join(clauses)

    def _limit_filter(
        self, max_rois: Optional[int] = None, max_tiles: Optional[int] = None, table_name: str = "ptv"
    ) -> str:
        assert max_rois is not None or max_tiles is not None, "At least one of max_rois or max_tiles must be provided"
        assert not (max_rois is not None and max_tiles is not None), "Only one of max_rois or max_tiles can be provided"

        if max_rois is not None:
            unique_rois = self.get_random_rois(max_rois)
            if isinstance(unique_rois, Sequence):
                filters = f"WHERE {table_name}.prepared_id IN {tuple(unique_rois)}"
            else:
                filters = f"WHERE {table_name}.prepared_id IN ({unique_rois})"
        else:
            if max_tiles is None:
                return ""
            elif max_tiles > 1:
                unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))
            elif max_tiles == 1:
                unique_rois, unique_tiles = self.get_random_tiles(max_tiles)

            if isinstance(unique_tiles, Sequence) and isinstance(unique_rois, Sequence):
                filters = (
                    f"WHERE {table_name}.prepared_id IN {tuple(unique_rois)} "
                    f"AND {table_name}.tile_name IN {tuple(unique_tiles)}"
                )
            else:
                filters = (
                    f"WHERE {table_name}.prepared_id IN ({unique_rois}) "
                    f"AND {table_name}.tile_name IN ({unique_tiles})"
                )
        return filters

    def _age_filter(self, hpfs: Sequence[int], table_name: str = "ptv") -> str:
        assert hpfs is not None, "hpfs must be provided"
        hpfs = tuple(hpfs) if len(hpfs) > 1 else f"({hpfs[0]})"
        return f"WHERE {table_name}.hpf IN {hpfs}"

    def _exists_filter(self, table_name_shortcut) -> str:
        if self.server_folder_path is None or str(self.server_folder_path).startswith("/clusterfs"):
            filters = f"WHERE {table_name_shortcut}.exists = TRUE"
        elif str(self.server_folder_path).startswith("/groups"):
            filters = f"WHERE {table_name_shortcut}.exists_prfs = TRUE"
        elif str(self.server_folder_path).startswith("/aws") or str(self.server_folder_path).startswith(
            "/workspace/CellObservatoryData"
        ):
            filters = f"WHERE {table_name_shortcut}.exists_aws = TRUE"
        elif str(self.server_folder_path).startswith("/lustre"):
            filters = f"WHERE {table_name_shortcut}.exists_oak = TRUE"
        else:
            raise ValueError(f"Unknown server_folder_path: {self.server_folder_path}")
        return filters

    def _synthetic_filter(self, table_name_shortcut) -> str:
        filters = f"WHERE {table_name_shortcut}.is_synthetic = TRUE"
        return filters

    def _has_annotations_filter(self, table_name_shortcut) -> str:
        filters = (
            " AND EXISTS ( SELECT 1 "
            f"FROM jsonb_each({table_name_shortcut}.pc_metadata_json::jsonb) AS e(k, v) "
            "WHERE (v -> 'mask_bbox_dict') IS NOT NULL AND (v -> 'mask_bbox_dict')::jsonb <> '{}'::jsonb)"
        )
        return filters

    def _filters_to_string(
        self,
        table_name: str,
        table_name_shortcut: str = "hc",
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        hpf_list: Optional[Sequence[int]] = None,
        roi_list: Optional[Sequence[int]] = None,
        tile_list: Optional[Sequence[str]] = None,
        timepoint_list: Optional[Iterable[int]] = None,
        synthetic_only: bool = False,
        has_annotations: bool = False,
    ) -> str:

        filters = self._exists_filter(table_name_shortcut)

        if synthetic_only:
            filters += self._synthetic_filter(table_name_shortcut).replace("WHERE", " AND ")

        if has_annotations:
            filters += self._has_annotations_filter(table_name_shortcut)

        if roi_list is not None or tile_list is not None or timepoint_list is not None:
            filters += self._choose_filter(
                rois=roi_list, tiles=tile_list, timepoints=timepoint_list, table_name=table_name_shortcut
            ).replace("WHERE", " AND ")
        elif max_rois is not None or max_tiles is not None:
            filters += self._limit_filter(
                max_rois=max_rois, max_tiles=max_tiles, table_name=table_name_shortcut
            ).replace("WHERE", " AND ")

        if hpf_list is not None:
            filters += self._age_filter(hpfs=hpf_list, table_name=table_name_shortcut).replace("WHERE", " AND ")

        if self.verbose:
            print(f"Using filters: {filters}")
        return filters

    def _query_t_128_128_128_2_hypercube_view(
        self,
        table_name: str,
        table_name_shortcut: str = "hc",
        num_timepoints: Optional[int] = 32,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Sequence[int]] = None,
        roi_list: Optional[Sequence[int]] = None,
        tile_list: Optional[Sequence[str]] = None,
        timepoint_list: Optional[Iterable[int]] = None,
        occupancy_threshold: Optional[float] = None,
        synthetic_only: bool = False,
        has_annotations: bool = False,
    ) -> list[str]:
        column_names = [
            "first_pc_id",
            "prepared_id",
            "tile_name",
            "x_start",
            "y_start",
            "z_start",
            "time_start",
            "channel_size",
            "cube_size",
            "time_size",
            "hpf",
            "server_folder",
            "output_folder",
            "pc_metadata_json",
            "p_metadata_json",
            "metadata_tile_json",
            "json_excite_map_total",
            "unique_targets",
            "imaged_locations",
            "date_crossed",
            "occupancy_ratios_ch_0",
            "occupancy_ratios_ch_1",
            "exists",
            "exists_prfs",
            "exists_aws",
            "is_synthetic",
        ]

        filters = self._filters_to_string(
            table_name=table_name,
            table_name_shortcut=table_name_shortcut,
            max_rois=max_rois,
            max_tiles=max_tiles,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list,
            timepoint_list=timepoint_list,
            synthetic_only=synthetic_only,
            has_annotations=has_annotations,
        )

        limit = f"LIMIT {max_hypercubes}" if max_hypercubes else ""

        return [
            f"""
                SELECT
                    {', '.join([f'{table_name_shortcut}.{col}' for col in column_names])}
                FROM {table_name} {table_name_shortcut}
                {filters} 
                ORDER BY first_pc_id DESC
                {limit}
            """
        ]

    def _create_t_128_128_128_2_hypercube_view(
        self,
        num_timepoints: Optional[int] = 1,
        max_hypercubes: Optional[int] = None,
    ) -> str:
        prepared_cubes_column_names = [
            "prepared_id",
            "tile_name",
            "x_start",
            "y_start",
            "z_start",
        ]
        prepared_tiles_view_column_names = [
            "hpf",
            "channel_size",
            "cube_size",
            "server_folder",
            "output_folder",
            "metadata_tile_json",
            "json_excite_map_total",
            "unique_targets",
            "imaged_locations",
            "date_crossed",
        ]

        return f"""
            SELECT 
                first_pc_id,
                time_start,
                time_size,
                {', '.join([f'{col}' for col in prepared_cubes_column_names + prepared_tiles_view_column_names])},
                occupancy_ratios[:{num_timepoints}] as occupancy_ratios_ch_0,
                occupancy_ratios[{num_timepoints} + 1:] as occupancy_ratios_ch_1,
                channel_targets,
                timepoints
            FROM
            (
                SELECT
                    min(pc.id) as first_pc_id,
                    {', '.join([f'pc.{col}' for col in prepared_cubes_column_names])},
                    {', '.join([f'ptv.{col}' for col in prepared_tiles_view_column_names])},
                    array_length(array_agg(pc.occupancy_ratio), 1) / ptv.channel_size as time_size,
                    div(pc.time::numeric, {num_timepoints}::numeric) * {num_timepoints}::numeric as time_start,
                    array_agg(pc.occupancy_ratio ORDER BY pc.channel, pc.time) as occupancy_ratios,
                    string_agg(pc.channel_target, ','::text ORDER BY pc.channel, pc.time) as channel_targets,
                    array_agg(pc.time ORDER BY pc.channel, pc.time) as timepoints
                FROM prepared_cubes pc
                JOIN prepared_tiles_view ptv
                ON (pc.prepared_id, pc.tile_name) = (ptv.prepared_id, ptv.tile_name)
                WHERE
                    ptv.cube_size = 128 AND ptv.channel_size = 2
                GROUP BY
                    {', '.join([f'pc.{col}' for col in prepared_cubes_column_names])},
                    {', '.join([f'ptv.{col}' for col in prepared_tiles_view_column_names])},
                    (div(pc.time::numeric, {num_timepoints}::numeric))
                {f"LIMIT {max_hypercubes}" if max_hypercubes is not None else ''}
            ) hypercubes
        """

    def save_hypercubes_dataframe(self, hypercubes_dataframe_path: Path):
        hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        hypercubes_dataframe_path.parent.mkdir(parents=True, exist_ok=True)

        if self.hypercubes_dataframe is not None:
            self.hypercubes_dataframe.to_csv(hypercubes_dataframe_path, index=True, header=True)
            print(f"Saved hypercubes dataframe to {hypercubes_dataframe_path}")
        else:
            raise ValueError("Cannot save hypercubes dataframe `self.hypercubes_dataframe` is empty.")

        configs = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and key != "hypercubes_dataframe":
                if isinstance(value, Path):
                    configs[key] = str(value)
                elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    try:
                        configs[key] = list(value) if value is not None else None
                    except TypeError:
                        configs[key] = str(value)
                else:
                    try:
                        ujson.dumps(value)
                        configs[key] = value
                    except (TypeError, ValueError):
                        configs[key] = str(value)

        with open(self.hypercubes_dataframe_path.with_suffix(".json"), "w") as f:
            ujson.dump(configs, f, indent=4, sort_keys=True, escape_forward_slashes=False)
        print(f"Saved hypercubes dataframe configs to {self.hypercubes_dataframe_path.with_suffix('.json')}")

    def execute_query(self, query: str | list[str]) -> pd.DataFrame:
        for i in range(3):
            try:
                # avoid the costly COUNT query for pandas by using arrow as an intermediate step
                # https://sfu-db.github.io/connector-x/freq_questions.html
                t0 = time.perf_counter()
                result = cx.read_sql(
                    conn=self._database_url + "?options=-c%20statement_timeout%3D600000",  # set timeout to 10min
                    query=query,
                    protocol=self.protocol,
                    return_type="arrow",
                )
                df = result.to_pandas(split_blocks=False, date_as_object=False)
                t1 = time.perf_counter()
                logger.info(f"Took {t1-t0:.2f} seconds to fetch dataframe with shape {df.shape}")
                return df

            except Exception as e:
                logger.warning(f"Attempt {i+1} failed with error: {e}. Retrying...")
        logger.error(f"Failed to execute query: {query}")
        raise

    def list_tables(self) -> Any:
        return self.execute_query("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")

    def list_views(self) -> Any:
        return self.execute_query("SELECT viewname FROM pg_views WHERE schemaname = 'public';")

    def get_table(self, table_name: str) -> Any:
        return self.execute_query(f"SELECT * FROM {table_name};")

    def get_columns(self, table_name: str) -> list[str]:
        return self.execute_query(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';"
        ).column_name.to_list()

    def count_rows(self, table_name: str) -> int:
        return self.execute_query(f"SELECT COUNT(*) FROM {table_name};").iloc[0, 0]

    def count_columns(self, table_name: str) -> int:
        return len(self.get_columns(table_name))

    def get_random_rois(self, num_rois: int = 1) -> list[int]:
        filter = self._exists_filter("prepared_tiles_view")

        if self.synthetic_only:
            filter += self._synthetic_filter("prepared_tiles_view").replace("WHERE", " AND ")

        if self.hpf_list is not None:
            filter += self._age_filter(hpfs=self.hpf_list, table_name="prepared_tiles_view").replace("WHERE", " AND ")
        if self.tile_list is not None:
            filter += self._choose_filter(tiles=self.tile_list, table_name="prepared_tiles_view").replace(
                "WHERE", " AND "
            )
        if self.roi_list is not None:
            filter += self._choose_filter(rois=self.roi_list, table_name="prepared_tiles_view").replace(
                "WHERE", " AND "
            )

        # FIXME: we need to ensure that each rank pulls the same random rois
        #        since otherwise different ranks may get different rois and
        #        then when we shard the dataset we actually don't shard the same
        #        dataframe leadning to possibly duplicated rois across ranks.
        # query = f"""
        #     -- Getting random ROIs
        #     SELECT DISTINCT prepared_id 
        #     FROM prepared_tiles_view {filter}
        #     LIMIT {num_rois}            
        # """  # ORDER BY random() could be slow on large tables
        query = f"""
            -- Getting deterministic subset of ROIs (ordered by prepared_id)
            SELECT DISTINCT prepared_id 
            FROM prepared_tiles_view {filter}
            ORDER BY prepared_id
            LIMIT {num_rois}
        """
        return self.execute_query(query).values.squeeze().tolist()

    def get_random_tiles(self, num_tiles: int = 1) -> list[tuple[int, str]]:
        filter = self._exists_filter("prepared_tiles_view")

        if self.synthetic_only:
            filter += self._synthetic_filter("prepared_tiles_view").replace("WHERE", " AND ")

        if self.hpf_list is not None:
            filter += self._age_filter(hpfs=self.hpf_list, table_name="prepared_tiles_view").replace("WHERE", " AND ")
        if self.tile_list is not None:
            filter += self._choose_filter(tiles=self.tile_list, table_name="prepared_tiles_view").replace(
                "WHERE", " AND "
            )
        if self.roi_list is not None:
            filter += self._choose_filter(rois=self.roi_list, table_name="prepared_tiles_view").replace(
                "WHERE", " AND "
            )

        # FIXME: we need to ensure that each rank pulls the same random tiles
        #        since otherwise different ranks may get different tiles and
        #        then when we shard the dataset we actually don't shard the same
        #        dataframe leadning to possibly duplicated tiles across ranks.
        # query = f"""
        #     -- Getting random tiles
        #     SELECT DISTINCT prepared_id, tile_name 
        #     FROM prepared_tiles_view {filter}
        #     LIMIT {num_tiles}
        # """  # ORDER BY random() could be slow on large tables
        query = f"""
            -- Getting deterministic subset of tiles (ordered by prepared_id, tile_name)
            SELECT DISTINCT prepared_id, tile_name 
            FROM prepared_tiles_view {filter}
            ORDER BY prepared_id, tile_name
            LIMIT {num_tiles}
        """
        return self.execute_query(query).values.squeeze().tolist()

    def check_view_exists(self, table_name: str) -> bool:
        query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE 
                -- table_schema = 'public' AND   -- commented out to allow for non-'public' schemas like in Trino
                table_name = '{table_name}'
            )
        """
        return self.execute_query(query).values.squeeze().tolist()

    def get_t_128_128_128_2_hypercubes(
        self,
        num_timepoints: Optional[int] = 1,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Sequence[int]] = None,
        roi_list: Optional[Sequence[int]] = None,
        tile_list: Optional[Sequence[str]] = None,
        timepoint_list: Optional[Iterable[int]] = None,
        hypercubes_dataframe_path: Optional[Path] = None,
        occupancy_threshold: Optional[float] = None,
        synthetic_only: bool = False,
        has_annotations: bool = False,
    ) -> pd.DataFrame:

        table_name = f"prepared_{num_timepoints}_128_128_128_2_hypercube_view"
        if self.check_view_exists(table_name):

            if self.verbose:
                print(f"Using table: {table_name} from the database named: {self.dbname}.")

            self.last_query = self._query_t_128_128_128_2_hypercube_view(
                table_name=table_name,
                num_timepoints=num_timepoints,
                max_rois=max_rois,
                max_tiles=max_tiles,
                max_hypercubes=max_hypercubes,
                hpf_list=hpf_list,
                roi_list=roi_list,
                tile_list=tile_list,
                timepoint_list=timepoint_list,
                occupancy_threshold=occupancy_threshold,
                synthetic_only=synthetic_only,
                has_annotations=has_annotations,
            )
        
        else:

            if self.verbose:
                print(f"Table: {table_name} not found in database: {self.dbname}. Creating a new view...")

            self.table_query = self._create_t_128_128_128_2_hypercube_view(
                num_timepoints=num_timepoints, max_hypercubes=max_hypercubes
            )

            self.last_query = self._query_t_128_128_128_2_hypercube_view(
                table_name=f"({self.table_query})",
                num_timepoints=num_timepoints,
                max_rois=max_rois,
                max_tiles=max_tiles,
                max_hypercubes=max_hypercubes,
                hpf_list=hpf_list,
                roi_list=roi_list,
                tile_list=tile_list,
                timepoint_list=timepoint_list,
                occupancy_threshold=occupancy_threshold,
                synthetic_only=synthetic_only,
                has_annotations=has_annotations,
            )

        if self.verbose:
            print(f"Executing query with protocol: {self.protocol}")
            print("\n".join(self.last_query) if isinstance(self.last_query, list) else self.last_query)

        table = self.execute_query(self.last_query)

        if "z_size" not in table.columns or "y_size" not in table.columns or "x_size" not in table.columns:
            table["z_size"] = self.z_slices
            table["y_size"] = self.y_slices
            table["x_size"] = self.x_slices

        if any(table["time_size"] != self.num_timepoints):
            print(f"`time_sizes` for all rows in the dataframe should be {self.num_timepoints}. Found: {table['time_size'].unique()}")
            table["time_size"] = self.num_timepoints

        # FIXME: is this the best default for has_annotations?
        if "has_annotations" not in table.columns:
            table["has_annotations"] = True if has_annotations else False

        num_rows, num_cols = table.shape
        print(f"\nRetrieved {num_rows} rows. \t Retrieved {num_cols} columns.")

        if hypercubes_dataframe_path is not None:
            self.save_hypercubes_dataframe(hypercubes_dataframe_path=hypercubes_dataframe_path)

        return table

    def update_data_locations(self, col="exists_aws", rows_filters=["2025/9%", "2025/10%"]) -> pd.DataFrame:
        filters = " OR ".join([f"output_folder LIKE '{f}'" for f in rows_filters])

        query = f"""
            UPDATE prepared
            SET {col} = True
            WHERE {filters}
        """
        self.execute_query(query)

        query = f"""
            SELECT id, output_folder, {col}
            FROM prepared
            WHERE {filters}
        """

        table = self.execute_query(query)
        num_rows, num_cols = table.shape
        print(table)
        print(f"\nUpdated {num_rows} rows using {filters=}.")
        return table

    def _aggregate(self, df_pd, group_cols, z_slices=128, y_slices=128, x_slices=128):
                        
        df = pl.from_pandas(df_pd)

        df = df.with_columns(
            pl.col("z_start").alias("cube_z_start"),
            pl.col("y_start").alias("cube_y_start"),
            pl.col("x_start").alias("cube_x_start"),
        )

        df = df.with_columns(
            (pl.col("z_start") // z_slices * z_slices).alias("z_start"),
            (pl.col("y_start") // y_slices * y_slices).alias("y_start"),
            (pl.col("x_start") // x_slices * x_slices).alias("x_start"),
        )

        def _merge_metadata_list(values):
            if values is None:
                return None
            if hasattr(values, "to_list"):
                values = values.to_list()
            if not isinstance(values, (list, tuple)):
                return None
            if len(values) == 0:
                return None

            merged = {}
            for entry in values:
                if entry is None:
                    continue
                if isinstance(entry, str):
                    try:
                        entry = ujson.loads(entry)
                    except Exception:
                        continue
                if not isinstance(entry, dict):
                    continue
                for key, val in entry.items():
                    existing = merged.get(key)
                    if isinstance(existing, dict) and isinstance(val, dict):
                        merged[key] = {**existing, **val}
                    else:
                        merged[key] = val
            return merged if merged else None

        def _merge_bbox_dict_list(values):
            """
            Merge a list of {cell_id -> bbox} dicts into a single dict,
            assuming all bboxes are in the same global coordinate system.
            Each bbox can be either:
            - [zmin, ymin, xmin, zmax, ymax, xmax], or
            - {"zmin": ..., "ymin": ..., ...}
            """
            if values is None:
                return None
            if hasattr(values, "to_list"):
                values = values.to_list()
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                return None

            merged = {}
            for entry in values:
                if entry is None:
                    continue
                if isinstance(entry, str):
                    try:
                        entry = ujson.loads(entry)
                    except Exception:
                        continue
                if not isinstance(entry, dict):
                    continue

                for cell_id, bbox in entry.items():
                    if bbox is None:
                        continue

                    if isinstance(bbox, (list, tuple)) and len(bbox) == 6:
                        # TODO: add support for different bbox formats
                        zmin, ymin, xmin, zmax, ymax, xmax = bbox
                    elif isinstance(bbox, dict):
                        zmin = bbox.get("zmin", float("inf"))
                        ymin = bbox.get("ymin", float("inf"))
                        xmin = bbox.get("xmin", float("inf"))
                        zmax = bbox.get("zmax", float("-inf"))
                        ymax = bbox.get("ymax", float("-inf"))
                        xmax = bbox.get("xmax", float("-inf"))
                    else:
                        continue

                    existing = merged.get(cell_id)
                    if existing is None:
                        merged[cell_id] = {
                            "zmin": zmin,
                            "ymin": ymin,
                            "xmin": xmin,
                            "zmax": zmax,
                            "ymax": ymax,
                            "xmax": xmax,
                        }
                    else:
                        merged[cell_id] = {
                            "zmin": min(existing.get("zmin", float("inf")), zmin),
                            "ymin": min(existing.get("ymin", float("inf")), ymin),
                            "xmin": min(existing.get("xmin", float("inf")), xmin),
                            "zmax": max(existing.get("zmax", float("-inf")), zmax),
                            "ymax": max(existing.get("ymax", float("-inf")), ymax),
                            "xmax": max(existing.get("xmax", float("-inf")), xmax),
                        }

            return merged if merged else None

        def _merge_histogram_list(values):
            if values is None:
                return None
            if hasattr(values, "to_list"):
                values = values.to_list()
            if not isinstance(values, (list, tuple)):
                return None
            if len(values) == 0:
                return None

            non_none = [v for v in values if v is not None]
            if len(non_none) == 0:
                return None

            if all(isinstance(v, dict) for v in non_none):
                merged = {}
                all_keys = set()
                for entry in non_none:
                    all_keys.update(entry.keys())

                for key in all_keys:
                    values_for_key = [entry.get(key) for entry in non_none if key in entry]
                    if values_for_key:
                        merged[key] = min(values_for_key)
                return merged if merged else None
            else:
                return non_none[0]

        if "pc_metadata_json" in df.columns:

            def _parse_json(s):
                if s is None:
                    return None
                try:
                    return ujson.loads(s)
                except Exception:
                    return None

            df = df.with_columns(
                pl.col("pc_metadata_json")
                .map_elements(_parse_json, return_dtype=pl.Object)
                .alias("_pc_metadata_parsed")
            )

            parsed_list = df.select(pl.col("_pc_metadata_parsed")).to_series().to_list()
            _channel_ids = set()
            for item in parsed_list:
                if isinstance(item, dict):
                    for k in item.keys():
                        _channel_ids.add(str(k))
            _channel_ids = sorted(_channel_ids)
        else:
            _channel_ids = []
            df = df.with_columns(pl.lit(None).alias("_pc_metadata_parsed"))

        if "pc_metadata_json" in df.columns:

            for ch in _channel_ids:

                def _get_ch(obj, ch_id=ch):
                    if not isinstance(obj, dict):
                        return None
                    ch_meta = obj.get(ch_id)
                    if ch_meta is None:
                        return None
                    return ujson.dumps(ch_meta)

                df = df.with_columns(
                    pl.col("_pc_metadata_parsed")
                    .map_elements(_get_ch, return_dtype=pl.Utf8)
                    .alias(f"pc_metadata_json_ch_{ch}")
                )

                def _get_histogram(ch_meta_str, ch_id=ch):
                    if ch_meta_str is None:
                        return None
                    if isinstance(ch_meta_str, str):
                        try:
                            ch_meta = ujson.loads(ch_meta_str)
                        except Exception:
                            return None
                    elif isinstance(ch_meta_str, dict):
                        ch_meta = ch_meta_str
                    else:
                        return None

                    hist = ch_meta.get("histogram")
                    if hist is None:
                        return None
                    return ujson.dumps(hist)

                def _get_mask_bbox_dict(row, ch_id=ch):
                    ch_meta_val = row[f"pc_metadata_json_ch_{ch}"]
                    if ch_meta_val is None:
                        return None
                    if isinstance(ch_meta_val, str):
                        try:
                            ch_meta = ujson.loads(ch_meta_val)
                        except Exception:
                            return None
                    elif isinstance(ch_meta_val, dict):
                        ch_meta = ch_meta_val
                    else:
                        return None

                    mask_bbox = ch_meta.get("mask_bbox_dict")
                    if not isinstance(mask_bbox, dict):
                        return None

                    z0 = row["cube_z_start"]
                    y0 = row["cube_y_start"]
                    x0 = row["cube_x_start"]

                    converted = {}
                    for cell_id, bbox in mask_bbox.items():
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 6:
                            zmin, ymin, xmin, zmax, ymax, xmax = bbox
                        elif isinstance(bbox, dict):
                            zmin = bbox.get("zmin")
                            ymin = bbox.get("ymin")
                            xmin = bbox.get("xmin")
                            zmax = bbox.get("zmax")
                            ymax = bbox.get("ymax")
                            xmax = bbox.get("xmax")
                        else:
                            continue

                        if None in (zmin, ymin, xmin, zmax, ymax, xmax):
                            continue

                        # shift into *global* coords for this cube
                        converted[cell_id] = {
                            "zmin": zmin + z0,
                            "ymin": ymin + y0,
                            "xmin": xmin + x0,
                            "zmax": zmax + z0,
                            "ymax": ymax + y0,
                            "xmax": xmax + x0,
                        }

                    return ujson.dumps(converted) if converted else None

                df = df.with_columns(
                    [
                        pl.col(f"pc_metadata_json_ch_{ch}")
                        .map_elements(_get_histogram, return_dtype=pl.Utf8)
                        .alias(f"histogram_ch_{ch}"),
                        pl.struct([f"pc_metadata_json_ch_{ch}", "cube_z_start", "cube_y_start", "cube_x_start"])
                        .map_elements(_get_mask_bbox_dict, return_dtype=pl.Utf8)
                        .alias(f"mask_bbox_dict_ch_{ch}"),
                    ]
                )
        else:
            _channel_ids = []
            df = df.with_columns(pl.lit(None).alias("_pc_metadata_parsed"))


        def _parse_string_col(expr: pl.Expr) -> pl.Expr:
            return (
                expr.cast(pl.Utf8)
                .str.strip_chars()
                .str.replace_all(r"^[\[\{\(]\s*", "", literal=False)
                .str.replace_all(r"\s*[\]\}\)]$", "", literal=False)
                .str.replace_all("\n", " ", literal=True)
                .str.replace_all('"', "", literal=True)
                .str.replace_all("'", "", literal=True)
                .str.replace_all(r"[,\s]+", " ", literal=False)
                .str.strip_chars()
                .str.extract_all(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
                .list.eval(pl.element().cast(pl.Float64))
            )

        def _parse_occupancy_expr(colname: str, dtypes: Dict[str, pl.DataType]) -> pl.Expr:
            dt = dtypes[colname]
            if isinstance(dt, pl.List) and dt.inner in (pl.Float32, pl.Float64):
                return pl.col(colname).cast(pl.List(pl.Float64))
            if dt == pl.Utf8:
                return _parse_string_col(pl.col(colname))
            raise TypeError(f"Unsupported dtype for {colname}: {dt!r}")

        df = df.with_columns(
            _parse_occupancy_expr("occupancy_ratios_ch_0", df.schema).alias("occ0"),
            _parse_occupancy_expr("occupancy_ratios_ch_1", df.schema).alias("occ1"),
        )

        T0 = int(df.select(pl.col("occ0").list.len().max()).item())
        T1 = int(df.select(pl.col("occ1").list.len().max()).item())

        occ0_mean_exprs = [pl.col("occ0").list.get(i).mean().alias(f"__occ0_{i}") for i in range(T0)]
        occ1_mean_exprs = [pl.col("occ1").list.get(i).mean().alias(f"__occ1_{i}") for i in range(T1)]

        occ0_concat = (
            pl.concat_list([pl.col(f"__occ0_{i}") for i in range(T0)])
            if T0 > 0
            else pl.lit([]).cast(pl.List(pl.Float64))
        )
        occ1_concat = (
            pl.concat_list([pl.col(f"__occ1_{i}") for i in range(T1)])
            if T1 > 0
            else pl.lit([]).cast(pl.List(pl.Float64))
        )

        pc_meta_list_exprs = []
        pc_metadata_full_expr = None
        if "pc_metadata_json" in df.columns:
            pc_meta_list_exprs.append(
                pl.col("pc_metadata_json").drop_nulls().implode().alias("__pc_metadata_full")
            )
            pc_metadata_full_expr = (
                pl.col("__pc_metadata_full")
                .map_elements(_merge_metadata_list, return_dtype=pl.Object)
                .map_elements(lambda d: ujson.dumps(d) if d is not None else None, return_dtype=pl.Utf8)
                .alias("pc_metadata_json")
            )

        pc_meta_ch_list_exprs = []
        for ch in _channel_ids:
            pc_meta_ch_list_exprs.append(
                pl.col(f"pc_metadata_json_ch_{ch}").drop_nulls().implode().alias(f"__pc_meta_list_ch_{ch}")
            )
        pc_meta_ch_concat_exprs = [
            pl.col(f"__pc_meta_list_ch_{ch}")
            .map_elements(_merge_metadata_list, return_dtype=pl.Object)
            .map_elements(lambda d: ujson.dumps(d) if d is not None else None, return_dtype=pl.Utf8)
            .alias(f"pc_metadata_json_ch_{ch}")
            for ch in _channel_ids
        ]

        histogram_list_exprs = []
        histogram_concat_exprs = []
        mask_bbox_dict_list_exprs = []
        mask_bbox_dict_concat_exprs = []

        for ch in _channel_ids:
            histogram_list_exprs.append(
                pl.col(f"histogram_ch_{ch}").drop_nulls().implode().alias(f"__histogram_list_ch_{ch}")
            )
            histogram_concat_exprs.append(
                pl.col(f"__histogram_list_ch_{ch}")
                .map_elements(_merge_histogram_list, return_dtype=pl.Object)
                .map_elements(lambda d: ujson.dumps(d) if d is not None else None, return_dtype=pl.Utf8)
                .alias(f"histogram_ch_{ch}")
            )

            mask_bbox_dict_list_exprs.append(
                pl.col(f"mask_bbox_dict_ch_{ch}").drop_nulls().implode().alias(f"__mask_bbox_dict_list_ch_{ch}")
            )
            mask_bbox_dict_concat_exprs.append(
                pl.col(f"__mask_bbox_dict_list_ch_{ch}")
                .map_elements(_merge_bbox_dict_list, return_dtype=pl.Object)
                .map_elements(lambda d: ujson.dumps(d) if d is not None else None, return_dtype=pl.Utf8)
                .alias(f"mask_bbox_dict_ch_{ch}")
            )

        agg_exprs = [
            pl.col("cube_size").first(),
            pl.col("time_size").first(),
            pl.col("channel_size").first(),
            pl.col("first_pc_id").first(),
            pl.col("hpf").first(),
            pl.col("server_folder").first(),
            pl.col("output_folder").first(),
            pl.col("unique_targets").first(),
            pl.col("imaged_locations").first(),
            pl.col("date_crossed").first(),
            pl.col("exists").max(),
            pl.col("exists_prfs").max(),
            pl.col("exists_aws").max(),
        ]

        # Only aggregate JSON columns if they actually exist
        if "p_metadata_json" in df.columns:
            agg_exprs.append(pl.col("p_metadata_json").sum())
        if "pc_metadata_json" in df.columns:
            agg_exprs.append(pl.col("pc_metadata_json").sum())
        if "metadata_tile_json" in df.columns:
            agg_exprs.append(pl.col("metadata_tile_json").sum())

        agg_exprs.extend(
            [
                *occ0_mean_exprs,
                *occ1_mean_exprs,
                *pc_meta_list_exprs,
                *pc_meta_ch_list_exprs,
                *histogram_list_exprs,
                *mask_bbox_dict_list_exprs,
            ]
        )

        out = (
            df.group_by(group_cols)
            .agg(agg_exprs)
            .with_columns(
                [occ0_concat.alias("occupancy_ratios_ch_0"), occ1_concat.alias("occupancy_ratios_ch_1")]
                + ([pc_metadata_full_expr] if pc_metadata_full_expr is not None else [])
                + pc_meta_ch_concat_exprs
                + histogram_concat_exprs
                + mask_bbox_dict_concat_exprs
            )
            .drop(
                [
                    *(f"__occ0_{i}" for i in range(T0)),
                    *(f"__occ1_{i}" for i in range(T1)),
                    *(["__pc_metadata_full"] if "pc_metadata_json" in df.columns else []),
                    *(f"__pc_meta_list_ch_{ch}" for ch in _channel_ids),
                    *(f"__histogram_list_ch_{ch}" for ch in _channel_ids),
                    *(f"__mask_bbox_dict_list_ch_{ch}" for ch in _channel_ids),
                ]
            )
        )

        # shift bbox back to local coordinates
        for ch in _channel_ids:
            colname = f"mask_bbox_dict_ch_{ch}"
            out = out.with_columns(
                pl.struct([colname, "z_start", "y_start", "x_start"])
                .map_elements(
                    lambda row, cname=colname: (
                        None
                        if row[cname] is None
                        else ujson.dumps(
                            {
                                cid: {
                                    "zmin": box["zmin"] - row["z_start"],
                                    "ymin": box["ymin"] - row["y_start"],
                                    "xmin": box["xmin"] - row["x_start"],
                                    "zmax": box["zmax"] - row["z_start"],
                                    "ymax": box["ymax"] - row["y_start"],
                                    "xmax": box["xmax"] - row["x_start"],
                                }
                                for cid, box in (
                                    ujson.loads(row[cname])
                                    if isinstance(row[cname], str)
                                    else {}
                                ).items()
                            }
                        )
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias(colname)
            )

        pdf = out.to_pandas()
        
        # FIXME: we should consider renaming these to generic names 
        #        for downstream processing
        def _get_col(pdf, base, channel):
            if channel is not None:
                name = f"{base}_ch_{channel}"
                if name in pdf.columns:
                    return pdf[name]
            return None

        if "mask_bbox_dict" not in pdf.columns:
            col = _get_col(pdf, "mask_bbox_dict", self.mask_channel)
            if col is not None:
                pdf["mask_bbox_dict"] = col

        return pdf

    def aggregate_hypercubes(
        self,
        z_slices: int = 128,
        y_slices: int = 128,
        x_slices: int = 128,
        group_cols: Iterable = ["time_start", "z_start", "y_start", "x_start", "prepared_id", "tile_name"],
    ):
        logger.info("Aggregating hypercubes...")
        t0 = time.perf_counter()
        self.hypercubes_dataframe = self._aggregate(
            self.hypercubes_dataframe, group_cols=group_cols, z_slices=z_slices, y_slices=y_slices, x_slices=x_slices
        )
        t1 = time.perf_counter()
        logger.info(f"Aggregated hypercubes in {t1 - t0:.2f} seconds.")

    def check_hypercube_sizes(
        self,
        df: pd.DataFrame,
        shape_df: pd.DataFrame,
        layout: str,
        join_keys: tuple[str, ...] = ("prepared_id",),
    ) -> pd.DataFrame:
        L = layout.upper()
        work = df.merge(
            shape_df[
                list(join_keys)
                + [
                    "tile_z_end",
                    "tile_y_end",
                    "tile_x_end",
                    "tile_x_start",
                    "tile_y_start",
                    "tile_z_start",
                    "tile_time_size",
                    "tile_channel_size",
                ]
            ],
            how="left",
            on=list(join_keys),
        )

        coord_map = {
            "T": {"start": "time_start", "size": "time_size"},
            "Z": {"start": "z_start", "size": "z_size"},
            "Y": {"start": "y_start", "size": "y_size"},
            "X": {"start": "x_start", "size": "x_size"},
            "C": {"start": None, "size": "channel_size"},
        }

        def _as_int(s):
            return pd.to_numeric(s).fillna(0).astype("int64")

        for col in [
            "time_start",
            "z_start",
            "z_size",
            "y_start",
            "y_size",
            "x_start",
            "x_size",
            "channel_size",
            "tile_z_end",
            "tile_y_end",
            "tile_x_end",
            "tile_x_start",
            "tile_y_start",
            "tile_z_start",
            "tile_time_size",
            "tile_channel_size",
        ]:
            work[col] = _as_int(work[col])

        axes_end_map = {
            "T": work["tile_time_size"],
            "Z": work["tile_z_end"] - work["tile_z_start"],
            "Y": work["tile_y_end"] - work["tile_y_start"],
            "X": work["tile_x_end"] - work["tile_x_start"],
            "C": work["tile_channel_size"],
        }

        # rows where any requested axis has no available extent (pure padding)
        valid_mask = pd.Series(True, index=work.index)

        for ax in L:
            meta = coord_map.get(ax)
            start_col, size_col = meta["start"], meta["size"]

            start = work[start_col] if start_col in work.columns else pd.Series(0, index=work.index, dtype="int64")
            req = work[size_col]

            end = axes_end_map[ax]
            available = np.clip(end - start, 0, None)
            effective_size = np.minimum(req, available)
            effective_series = pd.Series(effective_size, index=work.index)
            # mark rows with req>0 but effective_size==0 as invalid (pure padding)
            zero_and_positive_req = (effective_series == 0) & (req > 0)
            valid_mask &= ~zero_and_positive_req
            work[size_col] = effective_series.astype("int64")


        work = work[valid_mask].reset_index(drop=True)
        return work