import logging
from abc import ABC, abstractmethod
import PIL
from PIL import Image
import cv2
import numpy as np
from datasets import Dataset, DatasetDict

# Configure Logger:
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

class PreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess_image_dataset(self, dataset):
        """Abstract method to preprocess an image dataset"""
        pass

class HuggingFacePreprocessingStrategy(PreprocessingStrategy):

    def preprocess_image_dataset(self, dataset: Dataset) -> Dataset:
        """Abstract method to preprocess an image dataset"""
        res_dataset = None

        # Guard clauses
        if dataset is None:
            raise ValueError("Dataset can not be None")
        if not isinstance(dataset, DatasetDict) and not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a Dataset object, the dataset is type of: " + str(type(dataset)))
        if type(dataset["train"]["image"][0]) != Image.Image and type(dataset["train"]["image"][0]) != PIL.PngImagePlugin.PngImageFile:
            raise ValueError("Dataset must contain images, but is type of: " + str(type(dataset["train"]["image"][0])))

        # Cut out move boxes
        processed_img_dataset = self.generate_move_boxes_from_image_dataset(dataset)

        # Map image to label
        res_dataset = self.map_move_boxes_to_label(processed_img_dataset)

        return res_dataset

        ##TODO: Implement Method
    def generate_move_boxes_from_image_dataset(self, dataset: Dataset) -> Dataset:
        """Abstract method to generate move boxes from an image dataset"""
        ## Convert images from RGB to gray-scale
        dataset_gray_scale = self.process_image_dataset_rgb_to_grayscale(dataset)

        ## Convert gray-scale images to binary images with otsu's method
        dataset_binary_images = self.process_image_dataset_gray_scaled_to_binary_with_threshold(dataset_gray_scale)

        ## Generate image containing only grid lines
        list_of_image_with_grid_lines = self.generate_list_of_image_with_grid_lines(dataset_binary_images)

        ## Contour Algorithm
        """
        With this simplified image, we use a border following algorithm [25] to generate a hierarchical tree of contours. 
        Each contour is compressed into four points, storing only corners of each quadrilateral. Any contour which is 
        significantly larger or smaller than the size of a single move-box 
        (again, calculated relative to the total image size) can be ignored. The final contours are sorted based on 
        their positions relative to one another, and each is labeled by game, move, and player. Finally, we apply a 
        perspective correction to restore the original rectangular shape of the move-boxes and crop each of them with 
        a padding on the top and bottom of 15% and 25%, respectively, since written moves overflow the bottom of their 
        box more often and more severely than the top. This process is displayed in Figure 3. We did not pad box sides 
        because chess moves are short, and the players rarely need to cross the side boundaries. This method of 
        pre-processing is nearly agnostic to scoresheet style and will work with any scoresheet style, which includes 
        4-columns and solid grid lines.
        """

        return dataset_binary_images

    def process_image_dataset_rgb_to_grayscale(self, dataset: Dataset) -> Dataset:
        """
        Using the luminosity method: grayscale = 0.299 * R + 0.587 * G + 0.114 * B

        :param dataset: image dataset with images in RGB format and the corresponding text as target values
        :return: dataset with images in gray-scale format
        """
        dataset = dataset.map(lambda img: {"image": img["image"].convert("L")})

        return dataset

    def process_image_dataset_gray_scaled_to_binary_with_threshold(self, dataset: Dataset) -> Dataset:
        """
        Making the images in the dataset binary with Otsu's method

        :param dataset: image dataset with images in gray-scale format and the corresponding text as target values
        :return: dataset with images in binary format
        """
        try:
            dataset = dataset.map(lambda  img: {"image": self.process_image_gray_scaled_to_binary_with_threshold(img["image"])})
        except Exception as e:
            logger.error(f"Error during processing gray-scaled images to binary images: {str(e)}")
            raise

        logger.info(f"Complete Dataset with images in binary format processed successfully!")

        return dataset

    def process_image_gray_scaled_to_binary_with_threshold(self, image):
        """
        Uses Otsu's method to binarize the image.

        :param image: image in gray-scale format
        :return: image in binary format
        """
        if image is None:
            raise ValueError("Input image can not be None.")

        if not isinstance(image, Image.Image):
            raise ValueError(f"The input image must be of type PIL.Image, but is {type(image)}")

        try:
            np_image = np.array(image)
            ret, thresh = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_image = Image.fromarray(thresh)

            logger.debug(f"Image was converted to binary with the threshold of {ret}")
            return binary_image

        except Exception as e:
            logger.error(f"Error when converting to binary: {str(e)}")
            raise

    ##TODO: Implement Method
    def generate_list_of_image_with_grid_lines(self, listOfImages: list):
        res_img_list = []

        for img in listOfImages:
            res_img_list.append(self.generate_image_containing_only_grid_lines(img))

        return res_img_list

    ##TODO: Implement Method
    def generate_image_containing_only_grid_lines(self, image):
        """
        Hough Transform
        Use two long, thin kernels (one horizontal and one vertical) with sizes relative to input image dimensions,
        and morphological operations (erosion followed by dilation) with those kernels to generate an image containing
        only grid lines. - Digitization of Handwritten Chess Scoresheets with a BiLSTM Network

        :param image:
        :return:
        """




    ##TODO: Implement Method
    def generate_contour_image(self, image):
        res_img = None

        return res_img

        ##TODO: Implement Method
    def map_move_boxes_to_label(self, dataset: Dataset) -> Dataset:
        """Abstract method to map move boxes to a label"""
        return dataset