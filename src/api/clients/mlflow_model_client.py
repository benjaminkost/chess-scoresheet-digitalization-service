import logging
import os
from typing import Any

import requests
from torch import Tensor

# configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def predict(pixel_values: Tensor) -> Any | None:
    """
    Calling the endpoint of the mlflow container /invocations to get a prediction from model

    :param pixel_values: Torch.Tensor that is the tokenized image for inference.
    :return: json (dict) for the prediction.
    """
    # Setup url for mlflow prediction
    mlflow_docker_controller_base_url = os.getenv("MLFLOW_DOCKER_BASE_URL")
    prediction_url = f"{mlflow_docker_controller_base_url}/invocations"

    # Call mlflow endpoint
    try:
        prediction = requests.post(prediction_url, data=pixel_values)
        return prediction.json()
    except requests.RequestException as e:
        logger.error(f"Error when calling the /invocations endpoint of mlflow container: {e}")


def check_health() -> bool:
    """
    Calling the endpoint of the mlflow container /health to get a health status from container

    :return: True if health, False otherwise.
    """
    # Setup url for mlflow prediction
    mlflow_docker_controller_base_url = os.getenv("MLFLOW_DOCKER_BASE_URL")
    health_url = f"{mlflow_docker_controller_base_url}/health"

    # Call mlflow endpoint
    try:
        health_status = requests.get(health_url).status_code
        return health_status == 200
    except requests.RequestException as e:
        logger.error(f"Error when calling the /invocation endpoint of mlflow container: {e}")
        return False