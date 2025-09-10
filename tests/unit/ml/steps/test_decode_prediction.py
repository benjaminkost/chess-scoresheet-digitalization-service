import logging
from unittest.mock import Mock

import pytest
import torch
from transformers import ProcessorMixin

from src.ml.steps.decode_prediction import decode_prediction

def test_decode_prediction(mocker, caplog):
    mock_processor = mocker.Mock()
    mock_prediction = mocker.Mock()
    mock_return_value = ["test worked"]
    mock_processor.batch_decode.return_value = mock_return_value
    spy_log = caplog.at_level(logging.INFO)
    result = decode_prediction(processor=mock_processor, prediction=mock_prediction)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == "Decode ids returned by the model... "
    assert result == mock_return_value[0]