import logging
from abc import ABC, abstractmethod
from datasets import Dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for Data Splitting Strategy
# -----------------------------------------------
# This class defines a common interface for different data splitting strategies.
# Subclasses must implement the split_data method.
class DataSplittingStrategy(ABC):
    @abstractmethod
    def data_split(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        """
        Abstract class to split the dataset into training and testing set

        :param feature_column: the column where the image for the corresponding text is inside
        :param split: huggingface by default creates the split "train"
        :param dataset: huggingface dataset with images and the corresponding text as target values
        :param target_column: the column where the text for the corresponding image is inside
        :return: X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass

# Concrete Strategy for simple Train-Test Split
# ---------------------------------------------
# This strategy implements a simple train-test split.
class SimpleDataSplittingStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: The seed used by the random number generator.
        """
        self.test_size=test_size
        self.random_state=random_state

    def data_split(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        """
        Implements a simple data_split

        :param feature_column: the column where the image for the corresponding text is inside
        :param split: huggingface by default creates the split "train"
        :param dataset: hugging face dataset with images and the corresponding text as target values
        :param target_column: the column where the text for the corresponding image is inside
        :return: X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """

        logging.info("Performing simple train-test split.")

        # Define feature and target values
        X = dataset[split][feature_column]
        y = dataset[split][target_column]

        # Split
        X_train, y_train, X_test, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")

        return X_train, y_train, X_test, y_test

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