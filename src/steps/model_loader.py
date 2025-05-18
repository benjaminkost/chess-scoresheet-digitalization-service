import mlflow
from zenml import step


@step
def load_model(model_name: str):
    """

    :return: loaded a model
    """

    # Get the latest version for the model
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(name=model_name)[0].version

    # Construct the model URI
    model_uri = f'models:/{model_name}/{model_version}'

    # Load the model
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    return model