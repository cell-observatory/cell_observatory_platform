import os
import logging
import sqlite3
import pandas as pd
import tensorstore as ts
from supabase import create_client, Client
from dotenv import load_dotenv

from .data_config import DataConfig
from .data_utils import index_mapper


class FishDatabase:
    """
    Access the preprocessed dataset and metadata.
    """
    def __init__(self, data_config: DataConfig = None,
                 force_create_db = False,
                 clean_up_db = False,
                 exists="true"):
        if data_config is None:
            data_config = DataConfig()

        self.data_config = data_config
        self.force_create_db = force_create_db
        self.clean_up_db = clean_up_db

        # Load environment variables from .env file
        load_dotenv()

        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")

        assert url, f"Environment variable 'SUPABASE_URL' is unset or is empty. A local .env file could contain 'SUPABASE_URL=https://XXXXXXXXXXXXXXXXXXXX.supabase.co' and 'SUPABASE_KEY='"
        assert key, f"Environment variable 'SUPABASE_KEY' is not set or is empty. This could be a public key that you find from the 'connect' page on supabase."

        # connect to the database
        db: Client = create_client(url, key)

        # Query metadata
        self.metadata = (
                db.table("prepared")
                .select("acquisition_id", "created_at", "software_version", "output_folder", "exists")
                .eq("exists", exists)
                .execute()
                .data
        )

        # Metadata df, sorted using record creation time
        self.metadata = pd.DataFrame(self.metadata)
        self.metadata = self.metadata.sort_values(by='created_at')

        self._open_zarr_files()
        self._init_local_db()

    def _open_zarr_files(self):
        self.stores = []
        for i, output_folder in enumerate(self.metadata["output_folder"]):
            spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': output_folder}}
            try:
                store = ts.open(spec).result()
                self.stores.append(store)
            except Exception as e:
                logging.info(f'File does not exist: {output_folder}. Consider updating the database.')
                self.metadata['exists'].iloc[i] = False
        # only keep existing
        self.metadata = self.metadata[self.metadata['exists']]

    def _init_local_db(self):
        # local db name
        cwd = os.getcwd()
        local_db_name = os.path.join(cwd, repr(self.data_config) + ".db")
        self.local_db_name = local_db_name

        # check db exists before .connect since it would create the db if it didn't
        create_db = not os.path.isfile(local_db_name)
        self.con = sqlite3.connect(local_db_name)
        self.cur = self.con.cursor()

        if create_db or self.force_create_db:
            # if force creating db, we need to delete the existing one first
            if self.force_create_db: os.remove(local_db_name)

            # Create store_index_map table
            # TODO: add x0 and y0 table using chunk id for middle out cropping, t and z drop last is better
            cmd = """
            CREATE TABLE store_index_map (
                                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                                storeid INTEGER,
                                tile INTEGER,
                                t INTEGER,
                                z INTEGER,
                                y INTEGER,
                                x INTEGER,
                                c INTEGER
                            );
            """
            self.cur.execute(cmd)

            # Loop over each store to create index mapping
            # TODO: filter based on fill factor here
            for i, store in enumerate(self.stores):
                indices = index_mapper(store.shape, self.data_config)
                cmd = "INSERT INTO store_index_map(storeid, tile, t, z, y, x, c)  VALUES("+str(i)+",?, ?, ?, ?, ?, ?)"
                self.cur.executemany(cmd, indices)

        # update dataset length
        res = self.cur.execute("SELECT COUNT(*) FROM store_index_map")
        self.length = res.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # look up corresponding indices (note SQL uses 1-based indexing for autoincremented row id)
        cmd = "SELECT * FROM store_index_map where rowid = " + str(index+1)
        res = self.cur.execute(cmd)
        rowid, storeid, tile, t, z, y, x, c = res.fetchone()
        store = self.stores[storeid]

        z1, z2 = z * self.data_config.z, (z + 1) * self.data_config.z
        y1, y2 = y * self.data_config.y, (y + 1) * self.data_config.y
        x1, x2 = x * self.data_config.x, (x + 1) * self.data_config.x
        c1, c2 = x * self.data_config.c, (c + 1) * self.data_config.c

        item = store[tile, t, z1:z2, y1:y2, x1:x2, c1:c2].read().result()
        return item

    def __del__(self):
        self.con.close()
        if self.clean_up_db:
            os.remove(self.local_db_name)