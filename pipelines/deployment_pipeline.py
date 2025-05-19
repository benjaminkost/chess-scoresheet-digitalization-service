from zenml import pipeline
from ..steps.dockerize_mlflow_model import dockerize_mlflow_model_with_python
from ..steps.push_docker_image_to_docker_hub_step import push_docker_image_to_docker_hub


@pipeline(enable_cache=False)
def deployment_pipeline(docker_image_name: str, model_name: str):
    # Create a Docker image out of the model
    dockerize_mlflow_model_with_python(docker_image_name, model_name)

    # Push Docker image to docker hub
    push_docker_image_to_docker_hub()