import click
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from src.pipelines.deployment_pipeline import continous_deployment_pipeline, inference_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import logging

# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # Zum Zurücksetzen der Farbe

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

@click.command()
@click.option(
    "--stop-service",
        is_flag=True,
        default=False,
        help="Stop the service before deploying"
)
def run_main(stop_service: bool):
    """
    Run the TrOCR model deployment

    :param stop_service:
    :return:
    """

    model_name = "microsoft/trocr-base-handwritten"

    if stop_service:
        # Get the MLflow model deployer stack component
        model_deployer = MLFlowDeploymentService.get_active_model_deployer()

        # Fetch existing services with same pipeline name, step name and model name
        existing_services = model_deployer.find_model_server(
            pipeline_name="continous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        return

    # Run the continous deployment pipeline
    continous_deployment_pipeline()

    # Get the active model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Run the inference pipeline
    inference_pipeline()

    logger.info("Now run \n "
                f"mlflow ui --backend-store-ui {get_tracking_uri()}\n"
                "To inspect your experiment runs within the MLflow UI."
                "You can find your runs tracked within the `continous_deployment_pipeline` pipeline."
                "Here you'll also be able to compare the two runs.")

    # Fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="continous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step"
    )

    if service[0]:
        logger.info(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )

if __name__ == "__main__":
    run_main()