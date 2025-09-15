import logging

import torch
from PIL import Image

from src.ml.steps.tokenize_image import tokenize_image


def test_tokenize_image(mocker, caplog):
    pixel_value = torch.ones((1, 3, 224, 224))
    spy_log = caplog.at_level(logging.INFO)
    mock_processor_cls = mocker.Mock()
    mock_processor = mocker.Mock()
    mock_output = mocker.Mock()
    mock_output.pixel_values = pixel_value
    mock_processor.return_value = mock_output
    mock_processor_cls.return_value = mock_processor
    dummy_image = Image.new("RGB", (10,10))

    result = tokenize_image(mock_processor, dummy_image)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"Tokenizing image with processor"
    assert result.shape == (1, 3, 224, 224)