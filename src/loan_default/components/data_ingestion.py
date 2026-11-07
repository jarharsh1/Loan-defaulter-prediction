"""Stage 1 - Data ingestion.

In a real system this would pull from S3 / a warehouse / an API. Here we copy
the versioned reference dataset into the ingestion artifact folder so every
downstream stage reads from a single, reproducible location.
"""

import shutil

import pandas as pd

from loan_default.entity.config_entity import DataIngestionConfig
from loan_default.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest(self) -> str:
        src, dst = self.config.source_file, self.config.local_data_file
        if not src.exists():
            raise FileNotFoundError(f"Source data not found: {src}")
        shutil.copy(src, dst)
        rows = len(pd.read_csv(dst))
        logger.info("Ingested %d rows from %s -> %s", rows, src, dst)
        return str(dst)
