import logging
from src.classes_for_steps.preprocessing_strategy import PreprocessingStrategy

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

class Preprocessing:
    def __init__(self, strategy: PreprocessingStrategy):
        """
        Represents a class responsible for initializing and holding a preprocessing
        strategy used for data handling or processing tasks.

        Attributes
        ----------
        _strategy : PreprocessingStrategy
            Holds the preprocessing strategy passed during the instantiation of
            the class to define specific data preprocessing behavior.

        Parameters
        ----------
        strategy : PreprocessingStrategy
            The preprocessing strategy to be used for handling data preprocessing
            logic within the class.

        """
        logging.info("Initializing preprocessing strategy.")

        self._strategy = strategy

    def set_strategy(self, strategy: PreprocessingStrategy):
        """
        Sets the preprocessing strategy to be used.

        This method allows the configuration of the preprocessing strategy by setting
        an instance of a class that implements the `PreprocessingStrategy` interface.
        The selected strategy will determine the behavior of preprocessing operations
        within any related system or workflow.

        :param strategy: The preprocessing strategy to configure. Must be an
            instance of a class implementing the `PreprocessingStrategy` interface.
        """
        logging.info("Switching preprocessing strategy.")

        self._strategy = strategy

    def preprocess_image_dataset(self, dataset):
        """Abstract method to preprocess an image dataset"""
        return self._strategy.preprocess_image_dataset(dataset)