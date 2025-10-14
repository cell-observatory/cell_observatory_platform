import sys
import logging
from typing import Any
import pandas as pd
import trino

from data.databases.base_database import ParentDatabase

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrinoDatabase(ParentDatabase):
    TRINO_HOST='trino-ocp.int.janelia.org'
    TRINO_USER='trino'
    TRINO_CATALOG='betzigvast'
    TRINO_SCHEMA='betzigdb/cellobservatory'
    TRINO_PORT=443

    def __init__(self, **kwargs):
        kwargs['max_partitions'] = 1  # using trino direct (not through connector-x) so just 1 partition is possible
        super().__init__(**kwargs)

    def _load_uri(self):
        uri = f'trino+https://{self.TRINO_USER}:password@{self.TRINO_HOST}:{self.TRINO_PORT}/{self.TRINO_CATALOG}'     # connection token
        uri = f'trino+https://{self.TRINO_HOST}:{self.TRINO_PORT}/{self.TRINO_CATALOG}/{self.TRINO_SCHEMA}?verify=false&user=trino'     # connection token

        assert uri is not None, "TRINO_URI_* environment variable not set"
        return uri
    
    def execute_query(self, query: str | list[str]) -> pd.DataFrame:
        
        # Convert list to string
        if isinstance(query, list):
            assert len(query) == 1, f"code does not handle trino with a batch of queries and given query has length of {len(query)}, max partitions {self.max_partitions}, query {" ".join(query.split())}."
            query = query[0]
        
        query=query.replace(";", "")  # trino does not like semicolons
        query=query.replace(".exists,", '."exists",')  # trino does not like exists without quotes
        

        conn = trino.dbapi.connect(
            host=self.TRINO_HOST,
            user=self.TRINO_USER,
            catalog=self.TRINO_CATALOG,
            http_scheme="https",
            schema=self.TRINO_SCHEMA
        )
        TRINO_PORT = conn.port  # will be port 443 for https
        cur = conn.cursor()
    
        try:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            # pandas:
            df = pd.DataFrame(rows, columns=columns)
            return df
        except Exception as e:
            normalized_query = " ".join(query.split())
            logger.error(f"Failed to execute query: {e}. Query was: {normalized_query}")
            raise

    def list_tables(self) -> Any:
        return self.execute_query(f'''
                                  SELECT table_name
                                    FROM information_schema.tables
                                    WHERE table_catalog = '{self.TRINO_CATALOG}'
                                    AND table_schema = '{self.TRINO_SCHEMA}'
                                    AND table_type = 'BASE TABLE'
                                  ''')

    def list_views(self) -> Any:
        return self.execute_query(f'''
                                  SELECT table_name
                                    FROM information_schema.views
                                    WHERE table_catalog = '{self.TRINO_CATALOG}'
                                    AND table_schema = '{self.TRINO_SCHEMA}'
                                  ''')