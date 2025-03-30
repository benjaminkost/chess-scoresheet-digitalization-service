import logging

from datasets import Dataset
from abc import ABC, abstractmethod

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    # ANSI Escape Code for white letters
    WHITE = "\033[37m"
    RESET = "\033[0m"  # Zum Zurücksetzen der Farbe

    # Logger configure
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Console-Handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Formatter with ANSI Escape Code for white letters
    formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(levelname)s - %(message)s{RESET}')
    handler.setFormatter(formatter)

    # Handler for Logger added
    logger.addHandler(handler)

    @abstractmethod
    def ingest_image_dataset_from_huggingface(self, owner: str, dataset_name: str) -> Dataset:
        """Abstract method to ingest a image dataset from huggingface"""
        pass