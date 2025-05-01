import logging
from abc import ABC, abstractmethod
import PIL
from PIL import Image
import cv2
import numpy as np
from datasets import Dataset, DatasetDict
from pythreshold.global_th import otsu_threshold

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
    def process_image_dataset_with_grayscale(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {"image": x["image"].convert("L")})

        return dataset

    def process_image_gray_scaled_to_binary_with_threshold(self, image):
        """

        :param image:
        :return:
        """
        threshold = otsu_threshold(image)

        res_image = image.point(lambda x: 0 if x < threshold else 255, '1')

        return res_image

    def process_image_dataset_gray_scaled_to_binary_with_threshold(self, dataset: Dataset) -> Dataset:
        """
        The idea of this method cames from https://ieeexplore.ieee.org/document/4310076

        :param dataset:
        :return:
        """

        dataset = dataset.map(lambda  example: {"image": self.process_image_gray_scaled_to_binary_with_threshold(example["image"])})

        return dataset

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
        dataset_gray_scale = self.process_image_dataset_with_grayscale(dataset)

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

    ##TODO: Implement Method
    def generate_image_containing_only_grid_lines(self, image):
        """
        Use two long, thin kernels (one horizontal and one vertical) with sizes relative to input image dimensions,
        and morphological operations (erosion followed by dilation) with those kernels to generate an image containing
        only grid lines. - Digitization of Handwritten Chess Scoresheets with a BiLSTM Network

        :param image:
        :return:
        """

        # Konvertiere PIL Image zu numpy array für OpenCV
        try:
            img_array = np.array(image).astype(np.uint8)
        except Exception as e:
            print("Error converting PIL Image to numpy array: " + str(e))
            return None

        logger.info("Image shape: " + str(img_array.shape))

        # Bestimme die Kernel-Größen relativ zur Bildgröße
        height, width = img_array.shape
        horizontal_size = width // 3
        vertical_size = height // 25

        logger.info("Horizontal size: " + str(horizontal_size) + ", Vertical size: " + str(vertical_size))

        # Horizontaler Kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

        # Vertikaler Kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        # Horizontale Linien extrahieren
        try:
            horizontal = cv2.erode(img_array, horizontal_kernel)
            horizontal = cv2.dilate(horizontal, horizontal_kernel)
        except Exception as e:
            print("Error during horizontal line extraction: " + str(e))
            return None

        # Vertikale Linien extrahieren
        vertical = cv2.erode(img_array, vertical_kernel)
        vertical = cv2.dilate(vertical, vertical_kernel)

        # Kombiniere horizontale und vertikale Linien
        grid_lines = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)

        # Konvertiere zurück zu PIL Image
        return Image.fromarray(grid_lines)

    ##TODO: Implement Method
    def generate_list_of_image_with_grid_lines(self, listOfImages: list):
        res_img_list = []

        for img in listOfImages:
            res_img_list.append(self.generate_image_containing_only_grid_lines(img))

        return res_img_list

    ##TODO: Implement Method
    def generate_contour_image(self, image):
        res_img = None

        return res_img

        ##TODO: Implement Method
    def map_move_boxes_to_label(self, dataset: Dataset) -> Dataset:
        """Abstract method to map move boxes to a label"""
        return dataset