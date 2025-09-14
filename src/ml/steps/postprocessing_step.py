import logging

from chess.pgn import Game
from src.ml.scripts_for_steps.postprocessing_strategy import PostprocessingStrategy

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


def postprocessing_prediction(list_of_predictions: list) -> Game:
    """
    Turn list of chess moves into Chess Game in PGN format

    :param list_of_predictions: List of chess moves given from model inference
    :return: Chess Game in PGN format
    """
    postprocessing_strategy = PostprocessingStrategy()

    chess_game = postprocessing_strategy.turn_list_of_text_into_pgn(list_of_predictions)

    logger.info("Turned list of predictions (chess moves) into chess game in PGN format")

    return chess_game