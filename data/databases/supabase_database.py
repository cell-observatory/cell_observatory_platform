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
