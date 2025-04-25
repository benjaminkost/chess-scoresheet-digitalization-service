import logging
from abc import ABC, abstractmethod
import datasets
from datasets import Dataset
from datasets.exceptions import DatasetNotFoundError

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

# Implement a concrete class for Image Data Ingestion
class ImageDataIngestorImpl(DataIngestor):
    def ingest_image_dataset_from_huggingface(self, owner: str, dataset_name: str) -> Dataset:
        """
        Loads an image dataset from Hugging Face.

        :param owner: The owner/organization of the dataset on Hugging Face.
        :param dataset_name: The name of the dataset.
        :return: A `Dataset` object if successful, otherwise `None`.
        """
        try:
            res_hf_dataset = datasets.load_dataset(f"{owner}/{dataset_name}")
            self.logger.info(f"Dataset '{owner}/{dataset_name}' loaded successfully!")
            return res_hf_dataset
        except DatasetNotFoundError:
            self.logger.error(f"Dataset '{owner}/{dataset_name}' not found or inaccessible.")
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred while loading dataset '{owner}/{dataset_name}': {e}")