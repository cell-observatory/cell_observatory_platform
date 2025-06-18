import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional, Any, List, Tuple, Hashable, Sequence

import pandas as pd
from dotenv import load_dotenv
import connectorx as cx
from supabase import create_client, Client
from supabase.client import ClientOptions


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseDatabase:
    def __init__(
        self,
        dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
    ):
        self.dotenv_path = dotenv_path
        self.database_url = self._setup_url()
        self.client = self._setup_client()

    def _load_creds(self, dotenv_path: Path):
        assert Path(dotenv_path).exists(), f"{dotenv_path} was not found"

        load_dotenv(dotenv_path=dotenv_path, verbose=True)

        url = os.environ.get("SUPABASE_URL")
        assert url is not None, "SUPABASE_URL environment variable not set in .env file"

        key = os.environ.get("SUPABASE_KEY")
        assert key is not None, "SUPABASE_KEY environment variable not set in .env file"

        return url, key #re.escape(key)

    def _setup_url(self, port: int = 5432):
        url, key = self._load_creds(self.dotenv_path)
        return f"postgresql://postgres.{url}:{key}@aws-0-us-east-1.pooler.supabase.com:{port}/postgres"

    def _setup_client(self) -> Client:
        url, key = self._load_creds(self.dotenv_path)

        return create_client(
            supabase_url=f"https://{url}.supabase.co",
            supabase_key=key,
            options=ClientOptions(
                postgrest_client_timeout=60,
                storage_client_timeout=60,
                schema="public",
            )
        )

    def test_supabase_client(self) -> Any:
        return self.client.table("prepared").select("*").execute()

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            result = cx.read_sql(conn=self.database_url, query=query)
            logger.info(f"Query executed successfully. Returned {len(result)} rows.")
            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def test_connection(self) -> Any:
        return self.execute_query("SELECT * FROM prepared;")
