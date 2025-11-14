import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from cell_observatory_platform.data.databases.base_database import ParentDatabase


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseDatabase(ParentDatabase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_uri(self):

        if 'SUPABASE_STAGING_URI' not in os.environ or 'SUPABASE_PROD_URI' not in os.environ:
            assert Path(self.dotenv_path).exists(), f"{self.dotenv_path} was not found"
            if self.verbose:
                print(f"Loading additional environment variables from {self.dotenv_path}")
            load_dotenv(self.dotenv_path, verbose=True)

        if self.dbname == 'staging':
            uri = os.environ.get("SUPABASE_STAGING_URI")
        elif self.dbname == 'prod':
            uri = os.environ.get("SUPABASE_PROD_URI")
        else:
            raise ValueError(f"Unknown database name: {self.dbname}")

        assert uri is not None, "SUPABASE_URI_* environment variable not set"
        return uri
