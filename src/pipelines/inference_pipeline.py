from zenml import pipeline
from src.steps.predictor import predictor
from src.steps.dynamic_importer import dynamic_importer
from src.steps.model_loader import load_model

@pipeline(enable_cache=False)
def inference_pipeline(file_path: str, model_name: str):
    """
    Give a prediction for a given image

    :return: prediction
    """

    # Load image that was uploaded
    np_img = dynamic_importer(file_path)

    # Load the registered model
    model = load_model(model_name)

    # Run predictions on the image
    prediction_dict = predictor(model, np_img)

    return prediction_dict
