import logging

import chess.pgn

from src.ml.steps.decode_prediction import decode_prediction
from src.ml.steps.load_processor import load_processor
from src.ml.steps.postprocessing_step import postprocessing_prediction
from src.ml.steps.predict_preprocessing_step import preprocessing_image
from src.ml.steps.predictor import predictor
from src.ml.steps.dynamic_importer import load_png_image
from src.ml.steps.tokenize_image import tokenize_image

# configure logger
logging.basicConfig(
    level=logging.INFO,  # Log-Ebene (z. B. DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log-Format
)
logger = logging.getLogger(__name__)  # Logger mit Modulnamen beziehen

def inference_pipeline(image_file_path: str) -> chess.pgn.Game:
    """
    Give a prediction for a given image

    :return: prediction
    """

    # Load image that was uploaded
    logger.info(f"Loading image from {image_file_path}")
    input_image = load_png_image(image_file_path)

    # Preprocess image
    logger.info(f"Preprocessing image")
    list_of_move_boxes = preprocessing_image(input_image)

    # Load Processor to tokenize images
    logger.info(f"Load Processor")
    processor = load_processor()

    # Run the prediction
    list_of_predictions = []
    logger.info(f"Running prediction")
    for move_box in list_of_move_boxes:
        # convert images to RGB
        move_box = move_box.convert("RGB")

        # convert PIL image to torch.Tensor
        image_as_torch_tensor = tokenize_image(processor, move_box)

        # predict text on image
        prediction_ids = predictor(image_as_torch_tensor)
        prediction = decode_prediction(processor, prediction_ids)

        # Add it to the list of predictions for the move boxes
        list_of_predictions.append(prediction)

    # Post-process prediction list
    logger.info(f"Post-processing prediction")
    chess_game = postprocessing_prediction(list_of_predictions)

    return chess_game
