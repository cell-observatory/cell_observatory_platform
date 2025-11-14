import os
import sys
import logging
from typing import Any
import pandas as pd
import trino
from pathlib import Path
from dotenv import load_dotenv

from cell_observatory_platform.data.databases.base_database import ParentDatabase

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrinoDatabase(ParentDatabase):

    def __init__(self, **kwargs):          
        # TODO: update env vars when db is completely setup       
        self.trino_host = 'trino-ocp.int.janelia.org'
        self.trino_user = os.environ.get("TRINO_USER")
        self.trino_pass = os.environ.get("TRINO_PASS")
        self.trino_catalog = 'betzigvast'
        self.trino_schema = 'betzigdb/cellobservatory'
        self.trino_port = 443

        kwargs['max_partitions'] = 1  # using trino direct (not through connector-x) so just 1 partition is possible
        super().__init__(**kwargs)

    def execute_query(self, query: str | list[str]) -> pd.DataFrame:
        
        # Convert list to string
        if isinstance(query, list):
            assert len(query) == 1, f"code does not handle trino with a batch of queries and given query has length of {len(query)}, max partitions {self.max_partitions}, query {" ".join(query.split())}."
            query = query[0]
        
        query=query.replace(";", "")  # trino does not like semicolons
        query=query.replace(".exists,", '."exists",')  # trino does not like exists without quotes
        

        conn = trino.dbapi.connect(
            host=self.trino_host,
            user=self.trino_user,
            catalog=self.trino_catalog,
            http_scheme="https",
            schema=self.trino_schema,
            auth=trino.auth.BasicAuthentication(self.trino_user, self.trino_pass)
        )
        trino_port = conn.port  # will be port 443 for https
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
                                    WHERE table_catalog = '{self.trino_catalog}'
                                    AND table_schema = '{self.trino_schema}'
                                    AND table_type = 'BASE TABLE'
                                  ''')

    def list_views(self) -> Any:
        return self.execute_query(f'''
                                  SELECT table_name
                                    FROM information_schema.views
                                    WHERE table_catalog = '{self.trino_catalog}'
                                    AND table_schema = '{self.trino_schema}'
                                  ''')