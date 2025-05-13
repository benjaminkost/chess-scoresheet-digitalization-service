import os

from zenml import pipeline
from src.steps.predictor import predictor
from src.steps.dynamic_importer import dynamic_importer
from src.steps.model_loader import model_and_processor_loader
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

from src.steps.prediction_service_loader import prediction_service_loader


@pipeline
def continous_deployment_pipeline():
    """
    Pull huggingface model from hub and deploy it as a service

    :return:
    """
    # pull huggingface model
    trained_model, processor = model_and_processor_loader("microsoft", "trocr-base-handwritten")

    # (Re)deploy the trained model and processor
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=processor)

@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Give a prediction for a given image

    :return:
    """

    # Load image that was uploaded
    file_paths = os.listdir("./data")
    for file_path in file_paths:
        if ".png" in file_path:
            image_path = os.path.join("./data", file_path)
            break
    image = dynamic_importer(image_path)

    # Load the deployed model service and processor service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continous_deployment_pipeline",
        step_name="mlflow_model_deployer_step"
    )
    processor_service = prediction_service_loader(
        pipeline_name="continous_deployment_pipeline",
        step_name="mlflow_model_deployer_step"
    )

    # Run predictions on the image
    predictor(model_service=model_deployment_service, processor_service=processor_service, image=image)