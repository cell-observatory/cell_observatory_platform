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


    def _choose_filter(
        self,
        rois: Optional[Iterable[int|str]] = None,
        tiles: Optional[Iterable[str]] = None,
        table_name: str = 'ptv'
    ) -> str:

        assert rois is not None or tiles is not None, "At least one of rois or tiles must be provided"

        if rois is not None:
            rois = tuple(rois) if len(rois) > 1 else f"({rois[0]})"

        if tiles is not None:
            tiles = tuple(tiles) if len(tiles) > 1 else f"({tiles[0]})"

        if rois is not None and tiles is not None:
            return f"WHERE {table_name}.prepared_id IN {rois} AND {table_name}.tile_name IN {tiles}"
        elif rois is not None:
            return f"WHERE {table_name}.prepared_id IN {rois}"
        elif tiles is not None:
            return f"WHERE {table_name}.tile_name IN {tiles}"


    def _limit_filter(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        table_name: str = 'ptv'
    ) -> str:
        assert max_rois is not None or max_tiles is not None, "At least one of max_rois or max_tiles must be provided"

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
        return filters


    def _age_filter(
        self,
        hpfs: Iterable[int],
        table_name: str = 'ptv'
    ) -> str:
        assert hpfs is not None, "hpfs must be provided"

        hpfs = tuple(hpfs) if len(hpfs) > 1 else f"({hpfs[0]})"
        return f"WHERE {table_name}.hpf IN {hpfs}"


    def _filters_to_string(
        self,
        table_name: str,
        table_name_shortcut: str = 'hc',
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
    ) -> str:

        if roi_list is not None or tile_list is not None:
            filters = self._choose_filter(rois=roi_list, tiles=tile_list, table_name=table_name_shortcut)
        elif max_rois is not None or max_tiles is not None:
            filters = self._limit_filter(max_rois=max_rois, max_tiles=max_tiles, table_name=table_name_shortcut)
        else:
            filters = ''

        if hpf_list is not None:
            if filters == '':
                filters = self._age_filter(hpfs=hpf_list, table_name=table_name_shortcut)
            else:
                filters += self._age_filter(
                    hpfs=hpf_list, table_name=table_name_shortcut
                ).replace( 'WHERE', 'AND')

        logger.info(f"Using filters: {filters}")
        return filters

    def _query_hypercubes(
        self,
        table_name: str,
        table_name_shortcut: str = 'hc',
        num_timepoints: Optional[int] = 32,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
    ) -> str:
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
        ]

        filters = self._filters_to_string(
            table_name=table_name,
            table_name_shortcut=table_name_shortcut,
            max_rois=max_rois,
            max_tiles=max_tiles,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list,
        )

        return f"""
            SELECT
                {', '.join([f'{table_name_shortcut}.{col}' for col in column_names])},
                {table_name_shortcut}.occupancy_ratios[:{num_timepoints}] as occupancy_ratios_ch_0,
                {table_name_shortcut}.occupancy_ratios[{num_timepoints}+1:] as occupancy_ratios_ch_1,
                {table_name_shortcut}.timepoints[:{num_timepoints}] as timepoints_ch_0,
                {table_name_shortcut}.timepoints[{num_timepoints}+1:] as timepoints_ch_1
            FROM {table_name} {table_name_shortcut}
            {filters} 
            {f"LIMIT {max_hypercubes}" if max_hypercubes is not None else ''}
        """


    def get_32_128_128_128_2_hypercubes(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:

        self.last_query = self._query_hypercubes(
            table_name='prepared_32_128_128_128_2_hypercube_view',
            num_timepoints=32,
            max_rois=max_rois,
            max_tiles=max_tiles,
            max_hypercubes=max_hypercubes,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list
        )

        logger.info(f"Executing query: {self.last_query}")
        table = self.execute_query(self.last_query)
        return table


    def create_multichannel_hypercube_table(
        self,
        num_timepoints: Optional[int] = 1,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        max_hypercubes: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:

        prepared_cubes_column_names = [
            'prepared_id',
            'tile_name',
            'x_start',
            'y_start',
            'z_start',
        ]
        prepared_tiles_view_column_names = [
            'hpf',
            'channel_size',
            'cube_size',
            'server_folder',
            'output_folder',
            'metadata_json',
            'metadata_tile_json',
        ]

        self.table_query = f"""
            SELECT
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
        """

        self.last_query = self._query_hypercubes(
            table_name=f"({self.table_query})",
            num_timepoints=num_timepoints,
            max_rois=max_rois,
            max_tiles=max_tiles,
            max_hypercubes=max_hypercubes,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list
        )

        logger.info(f"Executing query: {self.last_query}")
        table = self.execute_query(self.last_query)
        return table

