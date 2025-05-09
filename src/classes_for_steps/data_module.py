import logging
from abc import ABC, abstractmethod
from datasets import Dataset

from src.classes_for_steps.data_loader_strategy import DataLoaderStrategy
from src.classes_for_steps.data_splitter_strategy import DataSplittingStrategy
from src.classes_for_steps.ingest_data_strategy import DataIngestorStrategy
from src.classes_for_steps.preprocessing_strategy import PreprocessingStrategy

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

# Define an abstract class for Data Ingestor
class DataModule(ABC):

    @abstractmethod
    def set_ingest_data_strategy(self, ingest_data_strategy: DataIngestorStrategy):
        pass

    @abstractmethod
    def ingest_data(self, owner: str, dataset_name: str) -> Dataset:
        """Abstract method to ingest a image dataset from huggingface"""
        pass

    @abstractmethod
    def set_preprocessing_strategy(self, preprocessing_strategy: PreprocessingStrategy):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def set_data_splitter_strategy(self, data_splitter_strategy: DataSplittingStrategy):
        pass

    @abstractmethod
    def split_dataset(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        pass

    @abstractmethod
    def set_dataloader_strategy(self, dataloader_strategy: DataLoaderStrategy):
        """Abstract method to get dataloader"""
        pass

    @abstractmethod
    def load_data(self, dataset: Dataset):
        pass

# Implement a concrete class for Image Data Ingestion
class HuggingFaceImageModule(DataModule):
    def __init__(self, ingest_data_strategy: DataIngestorStrategy,
                 preprocessing_strategy: PreprocessingStrategy,
                 data_splitter_strategy: DataSplittingStrategy,
                 dataloader_strategy: DataLoaderStrategy):
        logging.info("Initializing strategies for DataModule.")

        self._ingest_data_strategy = ingest_data_strategy
        self._preprocessing_strategy = preprocessing_strategy
        self._data_splitter_strategy = data_splitter_strategy
        self._dataloader_strategy = dataloader_strategy

    # Setter
    def set_ingest_data_strategy(self, ingest_data_strategy: DataIngestorStrategy):
        logging.info("Switching ingesting data strategy.")
        self._ingest_data_strategy = ingest_data_strategy

    def set_preprocessing_strategy(self, preprocessing_strategy: PreprocessingStrategy):
        logging.info("Switching preprocessing strategy.")
        self._preprocessing_strategy = preprocessing_strategy

    def set_data_splitter_strategy(self, data_splitter_strategy: DataSplittingStrategy):
        logging.info("Switching data splitter strategy.")
        self._preprocessing_strategy = data_splitter_strategy

    def set_dataloader_strategy(self, dataload_strategy: DataLoaderStrategy):
        logging.info("Switching data loader strategy.")
        self._dataloader_strategy = dataload_strategy

    # Executer
    def ingest_data(self, owner: str, dataset_name: str) -> Dataset:
        return self._ingest_data_strategy.ingest_data(owner, dataset_name)

    def preprocess_data(self, data):
        return self._preprocessing_strategy.transform(data)

    def split_dataset(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        return self._data_splitter_strategy.data_split(dataset, split, feature_column, target_column)

    # Loading Data
    def load_data(self, dataset: Dataset):
        return self._dataloader_strategy.load_data(dataset)
