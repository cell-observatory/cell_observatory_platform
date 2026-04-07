# DEPRECATED: This database is not used anymore.

# import os
# from dotenv import load_dotenv
# import vastdb
# import duckdb
# from ibis import _


# # creds from .env in same folder
# load_dotenv()
# ENDPOINT = os.getenv("VASTDB_ENDPOINT")
# ACCESS   = os.getenv("VASTDB_ACCESS")
# SECRET   = os.getenv("VASTDB_SECRET")

# #
# #  using vastdb sdk to query data into duckdb for aggregate queries
# #  connect -> session -> transaction -> database -> schema -> table -> duckdb aggregation query
# #

# conn = duckdb.connect()

# session = vastdb.connect(
#     endpoint=ENDPOINT,
#     access=ACCESS,
#     secret=SECRET)


# with session.transaction() as tx:
#     bucket = "betzigdb"
#     schema = "cellobservatory"
#     s = tx.bucket(bucket).schema(schema)

#     # prepared: only cols needed + filter pushed down
#     prepared_tbl = s.table("prepared").select(
#         columns=["id", "cube_size", "channel_size"],
#         predicate=(_["cube_size"] == 128) & (_["channel_size"] == 2),
#     )

#     # prepared_tiles: only cols we’ll use
#     tiles_tbl = s.table("prepared_tiles").select(
#         columns=["prepared_id", "tile_name"]
#     )

#     # join in DuckDB
#     con = duckdb.connect()
#     con.register("prepared", prepared_tbl)
#     con.register("prepared_tiles", tiles_tbl)

#     res = con.execute("""
#       SELECT pt.prepared_id, pt.tile_name
#       FROM prepared p
#       JOIN prepared_tiles pt ON pt.prepared_id = p.id
#     """).arrow()

# print(res.read_pandas())
