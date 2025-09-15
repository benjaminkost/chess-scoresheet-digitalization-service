import logging
import torch
from PIL.Image import Image
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

def tokenize_image(processor: TrOCRProcessor, image: Image) -> torch.Tensor:
    """
    Preparing the image for a transformer model (like TrOCR) to be used for inference

    :param processor: Processor/Tokenizer object (like TrOCRProcessor)
    :param image: Image of that should be tokenized
    :return: Torch tensor of tokenized image
    """

    logger.info(f"Tokenizing image with processor")

    pixel_values = processor(image=image, return_tensors="pt").pixel_values

    return pixel_values