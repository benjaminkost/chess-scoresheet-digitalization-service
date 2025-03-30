import datasets
from datasets import Dataset
from datasets.exceptions import DatasetNotFoundError

from src.pipeline_classes.DataIngestor import DataIngestor

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

        return None  # Explicit return in case of failure
