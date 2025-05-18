import os.path
from sys import version_info
import mlflow
from zenml import pipeline

from src.mlflow_models.trocr_mlflow_model import HFTransformerImageModelWrapper
from src.steps.log_register_custom_model import log_register_custom_model

@pipeline
def log_and_register_model():
    MODELS_DIR = "models"

    central_uri = os.path.abspath("../mlruns")

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Set python version
    PYTHON_VERSION = "{major}.{minor}".format(major=version_info.major, minor=version_info.minor)

    # Define artifacts
    artifacts = {
        "hf_model" : "./mlflow_model_configs/mlflow_model_config.json",
        "path": MODELS_DIR
    }
    # conda enviroment
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": [
                    "aiofiles==24.1.0",
                    "click",
                    "fastapi==0.115.12",
                    "mlflow==2.21.2",
                    "mlflow_skinny==2.21.2",
                    "numpy==2.2.5",
                    "opencv_python==4.11.0.86",
                    "pandas==2.2.3",
                    "Pillow==11.2.1",
                    "scikit_learn==1.6.1",
                    "starlette==0.46.2",
                    "transformers==4.51.3",
                    "zenml==0.82.1"
                ]
            }
        ]
    }

    # instantiate the model wrapper
    model = HFTransformerImageModelWrapper()

    # Define registered model name
    registered_model_name = "trocr-base-handwritten-with-pre-and-post-processing"

    # Log Transformer
    log_register_custom_model(
            model=model,
            conda_env=conda_env,
            artifacts=artifacts,
            artifact_path=MODELS_DIR,
            registered_model_name=registered_model_name,
        )