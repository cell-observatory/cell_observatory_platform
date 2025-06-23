import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any, List, Literal, Iterable

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
            SELECT DISTINCT ON (prepared_id) prepared_id 
            FROM prepared_tiles_view 
            LIMIT {num_rois}
        """
        return self.execute_query(query).values.squeeze().tolist()

    def get_random_tiles(self, num_tiles: int = 1) -> List[int]:
        query = f"""
            SELECT DISTINCT ON (prepared_id, tile_name) prepared_id, tile_name 
            FROM prepared_tiles_view 
            LIMIT {num_tiles}
        """
        return self.execute_query(query).values.squeeze().tolist()

    def _choose_filter(
        self,
        rois: Optional[List[int|str]] = None,
        tiles: Optional[List[str]] = None,
        table_name: str = 'ptv'
    ) -> str:
        if rois is not None and tiles is not None:
            return f"WHERE {table_name}.prepared_id IN {rois} AND {table_name}.tile_name IN {tiles}"
        elif rois is not None:
            return f"WHERE {table_name}.prepared_id IN {rois}"
        elif tiles is not None:
            return f"WHERE AND {table_name}.tile_name IN {tiles}"
        else:
            return ''


    def _limit_filter(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        table_name: str = 'ptv'
    ) -> str:
        if max_rois is None and max_tiles is None:
            return ''
        else:
            if max_rois is not None:
                unique_rois = self.get_random_rois(max_rois)
                if isinstance(unique_rois, Iterable):
                    filters = f"WHERE {table_name}.prepared_id IN {unique_rois}"
                else:
                    filters = f"WHERE {table_name}.prepared_id IN ('{unique_rois}')"
            else:
                if max_tiles > 1:
                    unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))
                else:
                    unique_rois, unique_tiles = self.get_random_tiles(max_tiles)
                print(unique_rois, unique_tiles)


                if isinstance(unique_tiles, Iterable) and isinstance(unique_rois, Iterable):
                    filters = f"WHERE {table_name}.prepared_id IN {unique_rois} " \
                              f"AND {table_name}.tile_name IN {unique_tiles}"
                else:
                    filters =  f"WHERE {table_name}.prepared_id IN ('{unique_rois}') " \
                               f"AND {table_name}.tile_name IN ('{unique_tiles}')"

        logger.info(f"Using filters: {filters}")
        return filters


    def get_32_128_128_128_2_hypercubes(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        column_names = [
            'prepared_id',
            'tile_name',
            'x_start',
            'y_start',
            'z_start',
            'time_start',
            'channel_size',
            'cube_size',
            'time_size',
            'hpf',
            'server_folder',
            'output_folder',
            'metadata_json',
            'metadata_tile_json',
            'occupancy_ratios_ch_0',
            'occupancy_ratios_ch_1'
        ]

        self.last_query = f"""
            SELECT
                {', '.join([f'hc.{col}' for col in column_names])},
                hc.timepoints[:32] as timepoints_ch_0,
                hc.timepoints[32+1:] as timepoints_ch_1
            FROM prepared_32_128_128_128_2_hypercube_view hc
            {self._limit_filter(max_rois=max_rois, max_tiles=max_tiles, table_name='hc')} 
            {f"LIMIT {max_rows}" if max_rows is not None else ''}
        """

        logger.info(f"Executing query: {self.last_query}")
        table = self.execute_query(self.last_query)
        return table