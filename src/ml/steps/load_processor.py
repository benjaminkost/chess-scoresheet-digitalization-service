import logging
import os
from transformers import TrOCRProcessor

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


def load_processor():
    hf_uri = f"{os.getenv("HF_USER_NAME")}/{os.getenv("HF_PROCESSOR_NAME")}"
    processor = TrOCRProcessor.from_pretrained(hf_uri)

    logger.info(f"Tokenizing image with processor from huggingface '{hf_uri}'")

    return processor