import logging
from transformers import TrOCRProcessor

from src.ml.steps.load_processor import load_processor

def test_load_processor(mocker, caplog, get_test_data_path):
    real_processor = TrOCRProcessor.from_pretrained(f"{get_test_data_path}/processor")
    mock_processor = mocker.patch.object(TrOCRProcessor,
                                         "from_pretrained",
                                         return_value = real_processor)
    spy_log = caplog.at_level(logging.INFO)
    test_env_values = "test"
    mock_hf_uri = f"{test_env_values}/{test_env_values}"
    mock_env = mocker.patch("os.getenv")
    mock_env.return_value = test_env_values

    result = load_processor()

    exp_messages = [
        f"Loading processor from huggingface with URI: '{mock_hf_uri}'",
        f"Processor loaded from huggingface with URI: '{mock_hf_uri}'"
    ]

    actual_messages = [msg for msg in spy_log.args[0].messages if mock_hf_uri in msg]
    mock_processor.assert_called_once_with(mock_hf_uri)
    assert type(result) == TrOCRProcessor
    assert actual_messages[:2] == exp_messages
