import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any, List, Literal

import pandas as pd
from dotenv import load_dotenv
import connectorx as cx


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseDatabase:
    def __init__(
        self,
        dbname: Literal['staging', 'production'] = 'staging',
        dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
    ):
        self.dbname = dbname
        self.dotenv_path = dotenv_path
        self.database_url = self._load_uri()

    def _load_uri(self):

        if 'SUPABASE_STAGING_URI' not in os.environ or 'SUPABASE_PROD_URI' not in os.environ:
            assert Path(self.dotenv_path).exists(), f"{self.dotenv_path} was not found"
            logger.info(f"Loading additional environment variables from {self.dotenv_path}")
            load_dotenv(self.dotenv_path, verbose=True)

        if self.dbname == 'staging':
            uri = os.environ.get("SUPABASE_STAGING_URI")
        elif self.dbname == 'production':
            uri = os.environ.get("SUPABASE_PROD_URI")
        else:
            raise ValueError(f"Unknown database name: {self.dbname}")

        assert uri is not None, "SUPABASE_URI_* environment variable not set"
        return uri

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            result = cx.read_sql(conn=self.database_url, query=query)
            logger.info(f"Query executed successfully. Returned {len(result)} rows.")
            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def list_tables(self) -> Any:
        return self.execute_query("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")

    def get_table(self, tablename: str) -> Any:
        return self.execute_query(f"SELECT * FROM {tablename};")

    def get_columns(self, tablename: str) -> List[str]:
        return self.execute_query(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tablename}';"
        ).column_name.to_list()

    def count_rows(self, tablename: str) -> int:
        return self.execute_query(f"SELECT COUNT(*) FROM {tablename};").iloc[0, 0]

    def count_columns(self, tablename: str) -> int:
        return len(self.get_columns(tablename))


    def get_random_rois(self, num_rois: int = 1) -> List[int]:
        query = f"""
            SELECT DISTINCT ON (prepared_id) prepared_id, tile_name 
            FROM prepared_tiles_view 
            LIMIT {num_rois}
        """
        return self.execute_query(query).values.tolist()

    def get_random_tiles(self, num_tiles: int = 1) -> List[int]:
        query = f"""
            SELECT DISTINCT ON (tile_name) prepared_id, tile_name 
            FROM prepared_tiles_view 
            LIMIT {num_tiles}
        """
        return self.execute_query(query).values.tolist()

    def get_random_rois_and_tiles(self, num_tiles: int = 1) -> List[int]:
        query = f"""
            SELECT DISTINCT ON (prepared_id, tile_name) prepared_id, tile_name 
            FROM prepared_tiles_view 
            LIMIT {num_tiles}
        """
        return self.execute_query(query).values.tolist()


    def sql_filter(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
    ) -> str:
        if max_rois is None and max_tiles is None:
            return ''
        else:
            if max_rois is not None and max_tiles is not None:
                unique_rois, unique_tiles = zip(*self.get_random_rois_and_tiles(max_tiles))
            elif max_rois is not None:
                unique_rois, unique_tiles = zip(*self.get_random_rois(max_rois))
            else:
                unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))

            if max_tiles > 1:
                filters = f"WHERE ptv.prepared_id IN {unique_rois} AND ptv.tile_name IN {unique_tiles}"
            else:
                filters = f"WHERE ptv.prepared_id IN ('{unique_rois[0]}') AND ptv.tile_name IN ('{unique_tiles[0]}')"

        return filters


    def get_3d_multichannel_hypercubes(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        prepared_cubes_column_names = [
            'prepared_id',
            'tile_name',
            'time',
            'x_start',
            'y_start',
            'z_start',
            'occupancy_ratio',
        ]
        prepared_tiles_view_column_names = [
            'channel_size',
            'cube_size',
            'time_size',
            'server_folder',
            'output_folder',
            'metadata_json',
            'metadata_tile_json',
        ]

        self.get_3d_multichannel_hypercubes_query = f"""
            SELECT
                {', '.join([f'pc.{col}' for col in prepared_cubes_column_names])},
                {', '.join([f'ptv.{col}' for col in prepared_tiles_view_column_names])},
                STRING_AGG(pc.channel::text, ',' ORDER BY pc.time, pc.channel) AS channels,
                STRING_AGG(pc.channel_target::text, ',' ORDER BY pc.time, pc.channel_target) AS channel_targets
            FROM prepared_cubes pc
            JOIN prepared_tiles_view ptv
            ON pc.prepared_id = ptv.prepared_id AND pc.tile_name = ptv.tile_name
            {self.sql_filter(max_rois=max_rois, max_tiles=max_tiles)} 
            GROUP BY
                {', '.join([f'pc.{col}' for col in prepared_cubes_column_names])},
                {', '.join([f'ptv.{col}' for col in prepared_tiles_view_column_names])}
            {f"LIMIT {max_rows}" if max_rows is not None else ''}
        """

        logger.info(f"Executing query: {self.get_3d_multichannel_hypercubes_query}")
        table = self.execute_query(self.get_3d_multichannel_hypercubes_query)
        table['channels'] = table['channels'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])
        table['channel_targets'] = table['channel_targets'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])

        return table


    def get_4d_multichannel_hypercubes(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        prepared_cubes_column_names = [
            'prepared_id',
            'tile_name',
            'x_start',
            'y_start',
            'z_start',
            'occupancy_ratio',
        ]
        prepared_tiles_view_column_names = [
            'channel_size',
            'cube_size',
            'time_size',
            'server_folder',
            'output_folder',
            'metadata_json',
            'metadata_tile_json',
        ]

        self.get_4d_multichannel_hypercubes_query = f"""
            SELECT
                {', '.join([f'pc.{col}' for col in prepared_cubes_column_names])},
                {', '.join([f'ptv.{col}' for col in prepared_tiles_view_column_names])},
                STRING_AGG(pc.time::text, ',' ORDER BY pc.time) AS timepoints,
                STRING_AGG(pc.channel::text, ',' ORDER BY pc.time, pc.channel) AS channels,
                STRING_AGG(pc.channel_target::text, ',' ORDER BY pc.time, pc.channel_target) AS channel_targets
            FROM prepared_cubes pc
            JOIN prepared_tiles_view ptv
            ON pc.prepared_id = ptv.prepared_id AND pc.tile_name = ptv.tile_name
            {self.sql_filter(max_rois=max_rois, max_tiles=max_tiles)} 
            GROUP BY
                {', '.join([f'pc.{col}' for col in prepared_cubes_column_names])},
                {', '.join([f'ptv.{col}' for col in prepared_tiles_view_column_names])}
            {f"LIMIT {max_rows}" if max_rows is not None else ''}
        """

        logger.info(f"Executing query: {self.get_4d_multichannel_hypercubes_query}")
        table = self.execute_query(self.get_4d_multichannel_hypercubes_query)

        table['timepoints'] = table['timepoints'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])
        table['channels'] = table['channels'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])
        table['channel_targets'] = table['channel_targets'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])

        return table
