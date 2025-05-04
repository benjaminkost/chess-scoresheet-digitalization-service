import logging
from abc import ABC, abstractmethod
import PIL
from PIL import Image
import cv2
import numpy as np
from datasets import Dataset, DatasetDict
from pandas.compat.numpy.function import SORT_DEFAULTS

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
        dataset_only_grid_lines = self.process_image_dataset_binary_to_grid_lines(dataset_binary_images)

        ## Contour Algorithm
        datatset_contour_list_per_image = self.generate_image_dataset_binary_grid_to_list_of_contours(dataset_only_grid_lines)

        ## Cut out boxes with padding, give them a name and map them to the corresponding label
        """
        The final contours are sorted based on their positions relative to one another(Done), and each 
        is labeled by game, move, and player. Finally, we apply a 
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
        dataset = dataset.map(lambda data_point: {"image": data_point["image"].convert("L")})

        return dataset

    def process_image_dataset_gray_scaled_to_binary_with_threshold(self, dataset: Dataset) -> Dataset:
        """
        Making the images in the dataset binary with Otsu's method

        :param dataset: image dataset with images in gray-scale format and the corresponding text as target values
        :return: dataset with images in binary format
        """
        try:
            dataset = dataset.map(lambda  data_point: {"image": self.process_image_gray_scaled_to_binary_with_threshold(data_point["image"])})
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

    def process_image_dataset_binary_to_grid_lines(self, dataset: Dataset) -> Dataset:
        """
        Using morphological operations to generate a binary image with only grid lines

        :param dataset: image dataset with images in binary format and the corresponding text as target values
        :return: binary image with only grid lines
        """
        try:
         dataset = dataset.map(lambda data_point: {"image": self.process_binary_image_to_grid_lines(data_point["image"])})
        except Exception as e:
            logger.error(f"Error during processing binary images to grid lines: {str(e)}")
            raise

        logger.info(f"Complete Dataset with images in grid lines processed successfully!")

        return dataset

    def process_binary_image_to_grid_lines(self, image):
        """
        Using morphological operations to generate a binary image with only grid lines

        :param image: image in binary format
        :return: binary image with only grid lines
        """

        # Guard clauses
        if image is None:
            raise ValueError("Input image can not be None.")

        # Convert to numpy
        np_img = np.array(image)

        # Invert image so lines are white
        np_img = cv2.bitwise_not(np_img)

        # Define kernel length
        horizontal_kernel_len = np_img.shape[1] // 40
        vertical_kernel_len = np_img.shape[0] // 40

        # Define kernel (matrix)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

        # Apply morphological operations with horizontal line
        ## Erosion
        horizontal_lines = cv2.erode(np_img, horizontal_kernel, iterations=1)
        ## Dilation
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

        # Apply morphological operations with vertical line
        ## Erosion
        vertical_lines = cv2.erode(np_img, vertical_kernel, iterations=1)
        ## Dilation
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

        # Bring processed images together
        grid_lines = cv2.add(horizontal_lines,vertical_lines)

        # Convert to PIL.Image
        res_img = Image.fromarray(grid_lines)

        return res_img

    def generate_image_dataset_binary_grid_to_list_of_contours(self, dataset: Dataset) -> Dataset:
        """
        Creates a list of contours per image, storing four point for the corners of each quadrilateral

        :param dataset: image dataset with images in binary format with grid lines and the corresponding text as target values
        :return: dataset with list of contours and the corresponding list of labels
        """

        # Replace each image with the list of the contours for the image
        dataset = dataset.map(lambda datapoint: {"image": self.generate_binary_grid_image_to_list_of_contours(datapoint["image"])})

        return dataset

    def generate_binary_grid_image_to_list_of_contours(self, image):
        """
        Creates a list of contours for the image. The contours store four points for the corners of each quadrilateral

        :param image: image in binary format with grid lines
        :return: list of contours
        """

        # Convert image to numpy
        np_img = np.array(image)

        # Find contours
        contours, hierarchy = cv2.findContours(np_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Compress each contour into four points, storing only corners of each quadrilateral
        simplified_contours = []
        for contour in contours:
            # Calculate perimeter of contour
            perimeter = cv2.arcLength(contour, True)

            # Approximate contour to rectangle
            ## Approximate contour to rectangle with a certain precision
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Add just contours with 4 points to simplified_contours list
            if len(approx) == 4:
                simplified_contours.append(approx)

        # Any contour which is significantly larger or smaller than the size of a single move-box can be ignored.
        median_area = np.median([cv2.contourArea(contour) for contour in simplified_contours])
        filtered_contours = [contour for contour in simplified_contours if 0.5 * median_area <= cv2.contourArea(contour) <= 1.5 * median_area]

        if len(filtered_contours) == 120:
            # Sort contours based on their positions relative to one another
            sorted_contours = self.get_contour_precendence(filtered_contours)
            return sorted_contours
        elif len(filtered_contours) < 120:
            raise ValueError(f"{len(filtered_contours)} are not enough contours found. Should be 120!")
        elif len(filtered_contours) > 120:
            raise ValueError(f"{len(filtered_contours)} ar too many contours found. Should be 120")
        return None

    def get_contour_precendence(self, contours):
        """
        Determine a sorted list of contours based on their positions relative to one another according to a chess scoresheet.

        :param contours: tuple of lists with 4 points for the corners of each quadrilateral
        :return: sorted list of contours based on their positions relative to one another according to a chess scoresheet
        """

        # Define focal point of each contour (move box)
        focal_points = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            focal_points.append((x + w / 2, y + h / 2, contour)) #contour

        # Normalize focal points so that clear columns and lines can be determined
        focal_points = self.normalize_focal_points(focal_points)

        # Sort focal points based on their x-coordinate
        sorted_focal_points = [item[2] for item in sorted(focal_points, key=lambda x: (x[1], x[0]))]
        res_contours = self.move_every_third_and_fourth_to_end(sorted_focal_points)

        return res_contours

    def normalize_focal_points(self, focal_points):
        # Calculate max x-Value
        max_x_value = max([x for x, y, contour in focal_points])
        # Calculate max y-Value
        max_y_value = max([y for x, y, contour in focal_points])

        # Calculate possible margin
        x_margin = max_x_value/300
        y_margin = max_y_value/400

        # Normalize focal points
        for index, focal_point_perspective in enumerate(focal_points):
            for index, focal_points_others in enumerate(focal_points):
                # Make x-Value of focal point the same if they are very close to another focal point
                if abs(focal_point_perspective[0] - focal_points_others[0]) <= x_margin:
                    tmp_list_of_tuple = list(focal_points[index])
                    tmp_list_of_tuple[0] = focal_point_perspective[0]
                    res_tuple = tuple(tmp_list_of_tuple)
                    focal_points[index] = res_tuple
                # Make y-Value of focal point the same if they are very close to another focal point
                if abs(focal_point_perspective[1] - focal_points_others[1]) <= y_margin:
                    tmp_list_of_tuple = list(focal_points[index])
                    tmp_list_of_tuple[1] = focal_point_perspective[1]
                    res_tuple = tuple(tmp_list_of_tuple)
                    focal_points[index] = res_tuple

        return focal_points

    def move_every_third_and_fourth_to_end(self, focal_points):
        result = []
        to_move = []

        for i, item in enumerate(focal_points):
            # Jeder 3. (Index 2, 6, 10, ...) oder 4. (Index 3, 7, 11, ...)
            if (i % 4 == 2) or (i % 4 == 3):
                to_move.append(item)
            else:
                result.append(item)

        result.extend(to_move)
        return result

    ##TODO: Implement Method
    def map_move_boxes_to_label(self, dataset: Dataset) -> Dataset:
        """Abstract method to map move boxes to a label"""
        return dataset