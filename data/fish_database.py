import os
import logging
import sqlite3
import numpy as np
import pandas as pd
import tensorstore as ts
from supabase import create_client, Client
from dotenv import load_dotenv

from .data_config import DataConfig, ColorMode
from .data_utils import index_mapper, middle_out_crop_start_index


class FishDatabase:
    """
    Access the preprocessed dataset and metadata.
    """
    def __init__(self, data_config: DataConfig = None,
                 force_create_db = False,
                 clean_up_db = False,
                 metadata = None,
                 dtype = np.uint16
                 ):
        if data_config is None:
            data_config = DataConfig()

        self.data_config = data_config
        self.force_create_db = force_create_db
        self.clean_up_db = clean_up_db

        # instantiate fields that will be populated later for book-keeping
        self.con = None
        self.cur = None
        self.local_db_name = None
        self.stores = []
        self.length = 0

        if metadata is None:
            metadata = self._query_remote_db()
        self.metadata = metadata

        # return if no data in database
        if len(self.metadata) == 0:
            return

        # check required metadata fields exist
        required_fields = ["created_at", "output_folder", "exists"]
        for field in required_fields:
            if field not in self.metadata:
                raise ValueError(f"Metadata required fields are missing: {required_fields}")

        # Metadata df, sorted using record creation time
        self.metadata = pd.DataFrame(self.metadata)
        self.metadata = self.metadata.sort_values(by='created_at')

        self._open_zarr_files()
        self._init_local_db()

    @staticmethod
    def _query_remote_db():
        # Load environment variables from .env file
        load_dotenv()
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        assert url, f"Environment variable 'SUPABASE_URL' is unset or is empty. A local .env file could contain 'SUPABASE_URL=https://XXXXXXXXXXXXXXXXXXXX.supabase.co' and 'SUPABASE_KEY='"
        assert key, f"Environment variable 'SUPABASE_KEY' is not set or is empty. This could be a public key that you find from the 'connect' page on supabase."
        # connect to the database
        db: Client = create_client(url, key)
        # Query metadata
        metadata = (
            db.table("prepared")
            .select("acquisition_id", "created_at", "software_version", "output_folder", "exists")
            .eq("exists", "true")
            .execute()
            .data
        )

        return metadata

    def _open_zarr_files(self):
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

            self._create_local_db_tables()

            # Loop over each store to create index mapping
            # TODO: filter based on fill factor here
            for i, store in enumerate(self.stores):
                indices = index_mapper(store.shape, self.data_config)
                cmd = "INSERT INTO store_index_map(storeid, tile, t, z, y, x, c)  VALUES("+str(i)+",?, ?, ?, ?, ?, ?)"
                self.cur.executemany(cmd, indices)

                y0, x0 = middle_out_crop_start_index(store.shape, self.data_config)
                cmd = "INSERT INTO middle_out_table(storeid, y0, x0)  VALUES(?, ?, ?)"
                self.cur.execute(cmd, (i, y0, x0))

        # update dataset length
        res = self.cur.execute("SELECT COUNT(*) FROM store_index_map")
        self.length = res.fetchone()[0]

    def _create_local_db_tables(self):
        # Create store_index_map table
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
        cmd = """
             CREATE TABLE middle_out_table (
                storeid INTEGER UNIQUE,
                y0 INTEGER,
                x0 INTEGER,
                FOREIGN KEY (storeid) REFERENCES store_index_map(storeid)
             );
             """
        self.cur.execute(cmd)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        # look up corresponding indices (note SQL uses 1-based indexing for autoincremented row id)
        cmd = "SELECT * FROM store_index_map WHERE rowid = ?"
        res = self.cur.execute(cmd, (index + 1,))
        resp = res.fetchone()
        if resp is None:
            return None
        rowid, storeid, tile, t, z, y, x, c = resp
        # retrieve store
        store = self.stores[storeid]

        # get pixel offset values for y and x
        cmd = "SELECT * FROM middle_out_table where storeid = ?"
        res = self.cur.execute(cmd, (storeid, ))
        resp = res.fetchone()
        if resp is None:
            return None
        _, y0, x0 = resp

        # compute index slices
        t1, t2 = z * self.data_config.t, (t + 1) * self.data_config.t
        z1, z2 = z * self.data_config.z, (z + 1) * self.data_config.z
        y1, y2 = y * self.data_config.y, (y + 1) * self.data_config.y
        x1, x2 = x * self.data_config.x, (x + 1) * self.data_config.x

        # slice data based on color mode
        if self.data_config.color_mode == ColorMode.MATCH:
            c1, c2 = x * self.data_config.c, (c + 1) * self.data_config.c
            item = store[tile, t1:t2, z1:z2, y1+y0:y2++y0, x1+x0:x2+x0, c1:c2].read().result()
        elif self.data_config.color_mode == ColorMode.AVG:
            item = store[tile, t1:t2, z1:z2, y1++y0:y2++y0, x1+x0:x2+x0, :].read().result()
            if item.shape[4] > 1:
                # cast to double (implicit) before averaging
                item = item.mean(4)
                # cast and reshape to original
                item = item.astype(self.dtype)[..., np.newaxis]
        else:
            raise NotImplemented("Color mode {self.data_config.color_mode} not implemented}")

        return item

    def __del__(self):
        if self.con:
            self.con.close()

        if self.clean_up_db and self.local_db_name is not None:
            try:
                os.remove(self.local_db_name)
            except Exception as e:
                logging.info(f'Failed to delete local db file: {self.local_db_name}.')
