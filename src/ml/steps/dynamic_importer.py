import logging
from pathlib import Path

from PIL import Image
import logging

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

def load_png_image(file_path: Path) -> Image.Image:
    """
    Loads the image file given by the path

    :param file_path: directory for an image file that was uploaded by a user
    :return: image as PIL.Image
    """

    logger.info(f"File from path '{file_path}' is about to get loaded...")

    if not file_path:
        raise Exception("No file path provided.")
    if not file_path.exists():
        raise FileExistsError(f"No file at path '{file_path}'")
    if not (file_path.suffix == ".png" or file_path.suffix == ".jpeg" or file_path.suffix == ".jpg"):
        raise TypeError("File path must point to a file with a file extension '.png', '.jpeg' or '.jpg'")

    image = Image.open(file_path)

    logger.info(f"Image from path '{file_path}' was successfully loaded")

    return image