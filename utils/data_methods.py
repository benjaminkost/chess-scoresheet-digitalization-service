import logging

import datasets
from datasets import Dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError

# Configure Logger:
# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # reset of color

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

def ingest_data(owner: str, dataset_name: str) -> None | Dataset | DatasetDict:
    """
    Loads an image dataset from Hugging Face.

    :param owner: The owner/organization of the dataset on Hugging Face.
    :param dataset_name: The name of the dataset.
    :return: A `Dataset` object if successful, otherwise `None`.
    """
    try:
        res_hf_dataset = datasets.load_dataset(f"{owner}/{dataset_name}")
        logger.info(f"Dataset '{owner}/{dataset_name}' loaded successfully!")
        return res_hf_dataset
    except DatasetNotFoundError:
        logger.error(f"Dataset '{owner}/{dataset_name}' not found or inaccessible.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading dataset '{owner}/{dataset_name}': {e}")