import os
import sys
import logging
from pathlib import Path

import ujson
import pandas as pd
from dotenv import load_dotenv
import connectorx as cx
import trino


TRINO_HOST='trino-ocp.int.janelia.org'
TRINO_USER='trino'
TRINO_CATALOG='betzigvast'
TRINO_SCHEMA='betzigdb/cellobservatory'
TRINO_PORT=443

conn = trino.dbapi.connect(
            host=TRINO_HOST,
            user=TRINO_USER,
            catalog=TRINO_CATALOG,
            http_scheme="https",
            schema=TRINO_SCHEMA
        )
TRINO_PORT = conn.port  # will be port 443 for https
cur = conn.cursor()

query= '''
--drop view if exists prepared_tiles_view;
create view prepared_tiles_view as
select distinct
	pc.prepared_id,
    pc.tile_name,
    p.server_folder,
    p.output_folder,
    p.acquisition_id,
    p."exists",
    p.cube_size,
    p.time_size,
    p.channel_size,
    p.metadata_json,
    p.data_location,
    json_query(p.metadata_json, 'strict $.channelPatterns') as channel_patterns,
    pt.metadata_tile_json,
    count_it.count as cube_count,
    gs.json_excite_map_total,
    gs."Unique Targets" as unique_targets,
    gs."Imaged Locations" as imaged_locations,
    gs."Date crossed" as date_crossed,
    gs.hpf as hpf
from
  prepared_tiles pt left join 
  prepared p on pt.prepared_id = p.id
  left join prepared_cubes pc on pc.prepared_id = pt.prepared_id and pt.tile_name = pc.tile_name
  join (
    select
      ppc.prepared_id,
      ppc.tile_name,
      count(*) as count
    from
      prepared_cubes ppc
    group by
      ppc.prepared_id,
      ppc.tile_name
  ) count_it on pc.prepared_id = count_it.prepared_id and pc.tile_name = count_it.tile_name
  left join g_sheet_master_imaging_list gs on gs."Data location" = p.data_location;
'''

try:
    cur.execute(query)
    rows = cur.fetchall()
    # pandas:
    df = pd.DataFrame(rows)
    return df
except Exception as e:
    normalized_query = " ".join(query.split())
    logger.error(f"Failed to execute query: {e}. Query was: {normalized_query}")
    raise