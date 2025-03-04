import os
import logging
import pandas as pd
import tensorstore as ts
from supabase import create_client, Client
from dotenv import load_dotenv

class FishDatabase:
    """
    Access the preprocessed dataset and metadata.
    """
    def __init__(self, exists="true"):
        # Load environment variables from .env file
        load_dotenv()

        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")

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

        self.metadata = pd.DataFrame(self.metadata)

        # Open zarr files
        self.stores = []
        for i, output_folder in enumerate(self.metadata["output_folder"]):
            spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': output_folder}}
            try:
                store = ts.open(spec).result()
                self.stores.append(store)
            except Exception as e:
                logging.info(f'File does not exist: {output_folder}. Consider updating the database.')
                self.metadata['exists'].iloc[i] = False

    def __len__(self):
        return len(self.stores)

    def __getitem__(self, index):
        return self.stores[index]