import logging

import torch
from transformers import ProcessorMixin, TrOCRProcessor

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


def decode_prediction(processor: TrOCRProcessor, prediction: torch.Tensor) -> str:
    """
    Decode the return of the transformer model (like TrOCR) which are ids

    :param processor: Processor/Tokenizer of the model
    :param prediction: Prediction/Ids returned by the model
    :return: Decoded prediction, which are normal characters
    """
    logger.info("Decode ids returned by the model... ")

    decoded_prediction = processor.batch_decode(prediction, skip_special_tokens=True)[0]

    return decoded_prediction


