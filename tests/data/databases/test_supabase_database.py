import pytest
from hydra.utils import instantiate
from hydra import initialize, compose
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")

@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../../configs/data/databases"):
        cfg = compose(config_name="supabase_database")
    return cfg


@pytest.fixture(scope="module")
def database(cfg):
    return instantiate(cfg)


def test_database_connection(database):
    tables = database.list_tables()
    assert tables is not None, "Connection to DB failed"
    print(f"Available tables: {tables.values.squeeze()}")


def test_all_database_tables(database):
    tables = database.list_tables()
    for t in tables.values.squeeze():
        cols = database.count_columns(t)
        rows = database.count_rows(t)

        print(f"Table `{t}` has {cols} column(s) and {rows} row(s).")
        try:
            assert cols > 0
        except AssertionError:
            print(f"Table `{t}` has no columns. Check if the table exists in the database.")

        try:
            assert rows > 0
        except AssertionError:
            print(f"Table `{t}` is empty. Check access to this table in the database.")


def test_table(database, tablename='prepared'):
    print(f"Testing table `{tablename}`...")
    cols = database.get_columns(tablename)
    num_cols = len(cols)
    num_rows = database.count_rows(tablename)

    assert num_cols > 1, f"Table `{tablename}` has {num_cols} column(s)"
    assert num_rows > 0, f"Table `{tablename}` has {num_rows} row(s)"
    print(f"Table `{tablename}` has {num_cols} column(s) and {num_rows} row(s).")
    pprint(cols)

def test_prepared_tiles_table(database):
    test_table(database, tablename='prepared_tiles')

def test_prepared_cubes_table(database):
    test_table(database, tablename='prepared_cubes')

def test_prepared_tiles_view_table(database):
    test_table(database, tablename='prepared_tiles_view')