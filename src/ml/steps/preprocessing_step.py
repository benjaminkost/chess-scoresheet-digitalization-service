import logging

from PIL import Image
from src.ml.scripts_for_steps.preprocessing_strategy import ThresholdMethod, HuggingFacePreprocessingStrategy

# Configure Logger:
# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # reset of color

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

def preprocessing_image(image: Image.Image) -> list:
    """
    Cutting out the move boxes from the given image of the chess scoresheet

    :param image: Image chess scoresheet
    :return: List of move boxes
    """
    # load preprocessing strategy
    ## Best parameters tested with tuning: src/tuning/preprocessing/preprocessing_hyperparameter_tuning.py
    kernelsize_gaussianBlur = (5, 5)
    sigmaX = 0
    threshold_method = ThresholdMethod.OTSU
    maxValue_threshold = 255
    block_size = 9
    c_value = 1
    horizontal_kernel_divisor = 30
    vertical_kernel_divisor = 20
    erosion_iterations = 1
    dilation_iterations = 1

    preprocessing_strategy = HuggingFacePreprocessingStrategy(
        kernelsize_gaussianBlur=kernelsize_gaussianBlur,
        sigmaX=sigmaX,
        threshold_method=threshold_method,
        maxValue_threshold=maxValue_threshold,
        block_size=block_size,
        c_value=c_value,
        horizontal_kernel_divisor=horizontal_kernel_divisor,
        vertical_kernel_divisor=vertical_kernel_divisor,
        erosion_iterations=erosion_iterations,
        dilation_iterations=dilation_iterations
    )

    logger.info(f"Preprocessing strategy initialized with parameters: {preprocessing_strategy.__dict__}")

    list_cut_out_move_boxes = preprocessing_strategy.preprocess_image(image)

    return list_cut_out_move_boxes