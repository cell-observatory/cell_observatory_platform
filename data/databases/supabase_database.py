import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any, List, Literal, Iterable

import ujson
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
        fetch_hypercubes_dataframe: bool = True,
        hypercubes_dataframe_path: Optional[Path] = None,
        use_cached_hypercubes_dataframe: Optional[bool] = False,
        protocol: Literal["binary", "csv", "cursor"] = "binary"
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
            hypercubes_dataframe_path: path to save hypercubes dataframe
            use_cached_hypercubes_dataframe: if True, will use the cached hypercubes dataframe from the given path
            protocol: The protocol used to fetch data from source (default is 'binary', can also be 'csv' or 'cursor')
            
        # TODO: Only works for `Tx128x128x128x2`, need to extend class to work with other hypercube sizes
        """

        if hypercubes_dataframe_path is None:
            self.hypercubes_dataframe_path = Path(__file__).parent.parent.parent / 'databases' / 'default_hypercubes_dataframe.csv'
        else:
            self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)

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
        self.use_cached_hypercubes_dataframe = use_cached_hypercubes_dataframe
        self.protocol = protocol

        self._database_url = self._load_uri()

        if self.fetch_hypercubes_dataframe:
            # if use_cached_hypercubes_dataframe is True, it is assumed that
            # hypercubes_dataframe_path points at a valid CSV file
            if self.use_cached_hypercubes_dataframe:
                self.hypercubes_dataframe = pd.read_csv(self.hypercubes_dataframe_path)
            else:
                self.hypercubes_dataframe = self.get_t_128_128_128_2_hypercubes(
                    num_timepoints=num_timepoints,
                    max_rois=max_rois,
                    max_tiles=max_tiles,
                    max_hypercubes=max_hypercubes,
                    hpf_list=hpf_list,
                    roi_list=roi_list,
                    tile_list=tile_list
                )

                # this is a hack to check that dataloading works correctly
                # remove when DB is updated fix the output_folder paths
                self.hypercubes_dataframe["output_folder"] = (
                    self.hypercubes_dataframe["output_folder"]
                        .str.replace(r"/6/9/", "/7/4/", regex=True)
                )

                self.save_hypercubes_dataframe(hypercubes_dataframe_path=self.hypercubes_dataframe_path)
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
                filters = f"WHERE {table_name}.prepared_id IN {tuple(unique_rois)}"
            else:
                filters = f"WHERE {table_name}.prepared_id IN ('{unique_rois}')"
        else:
            if max_tiles > 1:
                unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))
            else:
                unique_rois, unique_tiles = self.get_random_tiles(max_tiles)

            if isinstance(unique_tiles, Iterable) and isinstance(unique_rois, Iterable):
                filters = f"WHERE {table_name}.prepared_id IN {tuple(unique_rois)} " \
                          f"AND {table_name}.tile_name IN {tuple(unique_tiles)}"
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
                ).replace( 'WHERE', ' AND ')

        if self.verbose:
            print(f"Using filters: {filters}")
        return filters


    def _query_t_128_128_128_2_hypercube_view(
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
            'json_excite_map_total',
            'unique_targets',
            'imaged_locations',
            'date_crossed',
            'occupancy_ratios_ch_0',
            'occupancy_ratios_ch_1',
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
                {', '.join([f'{table_name_shortcut}.{col}' for col in column_names])}
            FROM {table_name} {table_name_shortcut}
            {filters} 
            {f"LIMIT {max_hypercubes}" if max_hypercubes is not None else ''}
        """


    def _create_t_128_128_128_2_hypercube_view(
        self,
        num_timepoints: Optional[int] = 1,
        max_hypercubes: Optional[int] = None,
    ) -> str:
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
            'json_excite_map_total',
            'unique_targets',
            'imaged_locations',
            'date_crossed',
        ]

        return f"""
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

    def save_hypercubes_dataframe(self, hypercubes_dataframe_path: Path):
        hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        hypercubes_dataframe_path.parent.mkdir(parents=True, exist_ok=True)

        if self.hypercubes_dataframe is not None:
            self.hypercubes_dataframe.to_csv(hypercubes_dataframe_path, index=True, header=True)
            print(f"Saved hypercubes dataframe to {hypercubes_dataframe_path}")
        else:
            raise ValueError('Cannot save hypercubes dataframe `self.hypercubes_dataframe` is empty.')

        configs = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'hypercubes_dataframe':
                if isinstance(value, Path):
                    configs[key] = str(value)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
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

        with open(self.hypercubes_dataframe_path.with_suffix('.json'), 'w') as f:
            ujson.dump(configs, f, indent=4, sort_keys=True, escape_forward_slashes=False)
        print(f"Saved hypercubes dataframe configs to {self.hypercubes_dataframe_path.with_suffix('.json')}")


    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            result = cx.read_sql(conn=self._database_url, query=query, protocol=self.protocol)
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