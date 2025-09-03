import os

from PIL.Image import Image
import requests


def predict(image: Image):
    # Call mlflow endpoint
    mlflow_docker_controller_base_url = os.getenv("MLFLOW_DOCKER_BASE_URL")
    prediction_url = f"{mlflow_docker_controller_base_url}/invocations"
    prediction = requests.post(prediction_url, data=image.tobytes())

    return prediction.json()