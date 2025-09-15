import io
import logging
import chess.pgn
from src.ml.scripts_for_steps.postprocessing_strategy import PostprocessingStrategy
from src.ml.steps.postprocessing_step import postprocessing_prediction


def test_postprocessing_prediction(mocker, caplog):
    pgn = io.StringIO("1. e4 e5 2. d4")
    mock_chess_game = chess.pgn.read_game(pgn)
    mocker.patch.object(PostprocessingStrategy,
                                           "turn_list_of_text_into_pgn",
                                           return_value=mock_chess_game)
    list_of_predictions = ["e4", "e5", "d4"]
    spy_log = caplog.at_level(logging.INFO)

    result = postprocessing_prediction(list_of_predictions)

    assert result == mock_chess_game
    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == "Turned list of predictions (chess moves) into chess game in PGN format"
