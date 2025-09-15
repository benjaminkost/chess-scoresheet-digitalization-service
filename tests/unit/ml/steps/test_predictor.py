import logging
import torch
from src.ml.steps.predictor import predict_chess_move


def test_predict_chess_move(mocker, caplog):
    mocker.patch("src.ml.steps.predictor.predict", return_value="e4")
    spy_log = caplog.at_level(logging.INFO)
    pixel_values = torch.tensor([[0, 0, 0], [1, 1, 1]])

    result = predict_chess_move(pixel_values)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"Prediction step starts..."
    assert result == "e4"