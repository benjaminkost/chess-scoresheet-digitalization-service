import logging

from PIL import Image

from src.ml.scripts_for_steps.preprocessing_strategy import HuggingFacePreprocessingStrategy
from src.ml.steps.preprocessing_step import preprocessing_image


def test_preprocessing_image(mocker, caplog):
    dummy_image = Image.new("RGB", (10, 10))
    mocker.patch.object(HuggingFacePreprocessingStrategy,
                                           "preprocess_image",
                                           return_value=[dummy_image])
    spy_log = caplog.at_level(logging.INFO)
    expected_result = [dummy_image]

    result = preprocessing_image(dummy_image)

    assert result == expected_result
    assert len(spy_log.args[0].messages) == 1