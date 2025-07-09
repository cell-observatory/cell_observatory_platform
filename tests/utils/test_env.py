import os
from pathlib import Path

from tests.conftest import config
from utils.container import get_container_info

import warnings
warnings.filterwarnings("ignore")


def test_env_vars(config):

    assert os.environ.get("SUPABASE_STAGING_URI") is not None, "SUPABASE_STAGING_URI is not set"

    assert os.environ.get("SUPABASE_PROD_URI") is not None, "SUPABASE_PROD_URI is not set"

    assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY is not set"

    assert os.environ.get("REPO_NAME") is not None, "REPO_NAME is not set"

    assert os.environ.get("REPO_DIR") is not None, "REPO_DIR is not set"

    assert os.environ.get("DATA_DIR") is not None, "DATA_DIR is not set"

    assert os.environ.get("STORAGE_SERVER_DIR") is not None, "STORAGE_SERVER_DIR is not set"



def test_env_container(config):

    container_info = get_container_info()
    print(f"ENV: {container_info}")

    assert container_info['container_type'] in ['native', 'docker', 'apptainer'], \
        f"Container type {container_info['container_type']} is not supported"
