import logging
from datasets import Dataset
from src.classes_for_steps.data_splitter_strategy import DataSplittingStrategy

# Configure Logger:
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

class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        :param strategy: The strategy to be used for data splitting.
        """
        logging.info("Initializing data splitting strategy.")
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        :param strategy: The new strategy to be used for data splitting.
        :return:
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        """
        Executes the data splitting using the current strategy.

        :param feature_column: the column where the image for the corresponding text is inside
        :param split: huggingface by default creates the split "train"
        :param dataset: hugging face dataset with images and the corresponding text as target values
        :param target_column: the column where the text for the corresponding image is inside
        :return: X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.data_split(dataset, split, feature_column, target_column)