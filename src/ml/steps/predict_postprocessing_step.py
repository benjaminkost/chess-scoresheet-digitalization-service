from chess.pgn import Game

from src.ml.scripts_for_steps.postprocessing_strategy import PostprocessingStrategy

def postprocessing_prediction(list_of_predictions: list) -> Game:
    postprocessing_strategy = PostprocessingStrategy()

    chess_game = postprocessing_strategy.turn_list_of_text_into_pgn(list_of_predictions)

    return chess_game