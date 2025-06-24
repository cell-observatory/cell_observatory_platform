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
        num_timepoints: Optional[int] = 1,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
        dbname: Literal['staging', 'production'] = 'staging',
        dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
        verbose: bool = False,
        fetch_hypercubes_dataframe: bool = True
    ):
        """
        A class for accessing Supabase database and retrieving hypercubes.
        The class is only setup to work with the a hypercube view of `Tx128x128x128x2` (TZYXC).
        It'll check database for existing views or create them if they don't exist based on the given `num_timepoints`.
        The results are stored in a pandas dataframe `self.hypercubes_dataframe`
        unless `fetch_hypercubes_dataframe` is set to False.

        Args:
            num_timepoints: number of timepoints for each hypercube
            max_rois: maximum number of ROIs (each ROI can have dozens of tiles)
            max_tiles: maximum number of tiles (each tile can have thousands of hypercubes)
            max_hypercubes: maximum number of hypercubes to return
            hpf_list: list of specific HPFs (hours-post-fertilization in hours) to filter
            roi_list: list of specific ROIs to filter
            tile_list: list of specific tiles to filter
            dbname: database name ('staging' or 'production')
            dotenv_path: path to .env file with URIs to access Supabase
            verbose: whether to print debug messages
            fetch_hypercubes_dataframe: this will automatically initialize the database based on the provided parameters
                (only turn off for debugging or if the database is already initialized)

        # TODO: Only works for `Tx128x128x128x2`, need to extend class to work with other hypercube sizes
        """
        self.dbname = dbname
        self.dotenv_path = dotenv_path
        self.num_timepoints = num_timepoints
        self.max_rois = max_rois
        self.max_tiles = max_tiles
        self.max_hypercubes = max_hypercubes
        self.hpf_list = hpf_list
        self.roi_list = roi_list
        self.tile_list = tile_list
        self.verbose = verbose
        self.fetch_hypercubes_dataframe = fetch_hypercubes_dataframe

        self.database_url = self._load_uri()

        if self.fetch_hypercubes_dataframe:
            self.hypercubes_dataframe = self.get_t_128_128_128_2_hypercubes(
                num_timepoints=num_timepoints,
                max_rois=max_rois,
                max_tiles=max_tiles,
                max_hypercubes=max_hypercubes,
                hpf_list=hpf_list,
                roi_list=roi_list,
                tile_list=tile_list
            )
        else:
            self.hypercubes_dataframe = None

    def _load_uri(self):

        if 'SUPABASE_STAGING_URI' not in os.environ or 'SUPABASE_PROD_URI' not in os.environ:
            assert Path(self.dotenv_path).exists(), f"{self.dotenv_path} was not found"
            if self.verbose:
                print(f"Loading additional environment variables from {self.dotenv_path}")
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
            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise


    def list_tables(self) -> Any:
        return self.execute_query("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")


    def list_views(self) -> Any:
        return self.execute_query("SELECT viewname FROM pg_views WHERE schemaname = 'public';")


    def get_table(self, table_name: str) -> Any:
        return self.execute_query(f"SELECT * FROM {table_name};")


    def get_columns(self, table_name: str) -> List[str]:
        return self.execute_query(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';"
        ).column_name.to_list()


    def count_rows(self, table_name: str) -> int:
        return self.execute_query(f"SELECT COUNT(*) FROM {table_name};").iloc[0, 0]


    def count_columns(self, table_name: str) -> int:
        return len(self.get_columns(table_name))


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


    def check_view_exists(self, table_name: str) -> bool:
        query =  f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = '{table_name}'
            )
        """
        return self.execute_query(query).values.squeeze().tolist()


    def get_t_128_128_128_2_hypercubes(
        self,
        num_timepoints: Optional[int] = 1,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
        hypercubes_dataframe_path: Optional[Path] = None
    ) -> pd.DataFrame:

        table_name = f'prepared_{num_timepoints}_128_128_128_2_hypercube_view'
        if self.check_view_exists(table_name):

            if self.verbose:
                print(f"Using table: {table_name} from the {self.dbname}.")

            self.last_query = self._query_t_128_128_128_2_hypercube_view(
                table_name=table_name,
                num_timepoints=num_timepoints,
                max_rois=max_rois,
                max_tiles=max_tiles,
                max_hypercubes=max_hypercubes,
                hpf_list=hpf_list,
                roi_list=roi_list,
                tile_list=tile_list
            )
        else:

            if self.verbose:
                print(f"Table: {table_name} not found in the {self.dbname}. Creating a new view...")

            self.table_query = self._create_t_128_128_128_2_hypercube_view(
                num_timepoints=num_timepoints,
                max_hypercubes=max_hypercubes
            )

            self.last_query = self._query_t_128_128_128_2_hypercube_view(
                table_name=f"({self.table_query})",
                num_timepoints=num_timepoints,
                max_rois=max_rois,
                max_tiles=max_tiles,
                max_hypercubes=max_hypercubes,
                hpf_list=hpf_list,
                roi_list=roi_list,
                tile_list=tile_list
            )

        if self.verbose:
            print(f"Executing query: {self.last_query}")

        table = self.execute_query(self.last_query)

        if hypercubes_dataframe_path is not None:
            self.save_hypercubes_dataframe(table, hypercubes_dataframe_path=hypercubes_dataframe_path)

        return table