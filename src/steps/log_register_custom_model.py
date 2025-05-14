import mlflow

def log_register_custom_model(model, conda_env, artifact_path, registered_model_name) -> None:
    with mlflow.start_run() as run:
        # log the Python function model
        mlflow.pyfunc.log_model(
            python_model=model,
            conda_env=conda_env,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )