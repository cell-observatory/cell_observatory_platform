import pytest
import sys
import logging
from hydra.utils import instantiate
from hydra import initialize, compose
from omegaconf import DictConfig

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../../configs/data/databases"):
        cfg = compose(config_name="supabase_database")
    return cfg


def test_supabase_database_connection(cfg: DictConfig):
    db = instantiate(cfg)
    assert db.client is not None, "Supabase client not initialized"

    res = db.test_supabase_client()
    print(res)
    assert res is not None, "Connection to Supabase client failed"

    res = db.test_connection()
    print(res)
    assert res is not None, "Connection to DB failed"
