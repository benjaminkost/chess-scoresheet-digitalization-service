import logging

import torch
from src.api.clients.mlflow_model_client import predict

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


def predict_chess_move(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Predict character from image with a model

    :param pixel_values: tokenize image
    :return: Ids from characters that were detected by the model
    """
    logger.info(f"Prediction step starts...")

    # Run inference
    prediction = predict(pixel_values)

    return prediction