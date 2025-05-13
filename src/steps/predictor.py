from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from PIL import Image

from src.steps.preprocessing_step import preprocess_image


@step(enable_cache=False)
def predictor(model_service: MLFlowDeploymentService, processor_service: MLFlowDeploymentService, image: Image) -> list:
    """
    Run an inference request against a deployed model.

    :param model_service: service where the model is deployed
    :param image: Image to be classified
    :return: the model prediction in np.ndarray format
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Image must be a PIL.Image.Image object.")

    # Start the service (should be a NOP if already started)
    model_service.start(timeout=10)
    processor_service.start(timeout=10)

    # Preprocess image
    list_of_move_boxes = preprocess_image(image)

    # Run the prediction
    list_of_predictions = []
    for move_box in list_of_move_boxes:
        prediction_encoded = model_service.predict(move_box)
        prediction = processor_service.predict(prediction_encoded)
        list_of_predictions.append(prediction)

    return list_of_predictions


