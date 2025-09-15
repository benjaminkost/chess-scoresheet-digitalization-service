import logging
import os
from dotenv import load_dotenv
from transformers import TrOCRProcessor

from src.ml.steps.load_processor import load_processor

def test_load_processor(mocker, caplog, get_test_data_path):
    real_processor = TrOCRProcessor.from_pretrained(f"{get_test_data_path}/processor")
    mock_processor = mocker.patch.object(TrOCRProcessor,
                                         "from_pretrained",
                                         return_value = real_processor)
    spy_log = caplog.at_level(logging.INFO)
    load_dotenv()
    hf_uri = f"{os.getenv("HF_USER_NAME")}/{os.getenv("HF_PROCESSOR_NAME")}"

    result = load_processor()

    mock_processor.assert_called_once_with(hf_uri)
    assert type(result) == TrOCRProcessor
    assert len(spy_log.args[0].messages) == 2
    assert spy_log.args[0].messages[0] == f"Loading processor from huggingface with URI: '{hf_uri}'"
    assert spy_log.args[0].messages[1] == f"Processor loaded from huggingface with URI: '{hf_uri}'"
