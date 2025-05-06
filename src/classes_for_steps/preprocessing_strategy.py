import logging
from abc import ABC, abstractmethod
import PIL
import datasets
from PIL import Image
import cv2
import numpy as np
from datasets import Dataset, DatasetDict, Features

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
        """
        Change a dataset of images and labels, so that the image is splitted into smaller images
        (containing the move boxes) and the list of labels is splitted into the label which is mapped to the smaller image

        :param dataset:
        :return:
        """

        # Guard clauses
        if dataset is None:
            raise ValueError("Dataset can not be None")
        if not isinstance(dataset, DatasetDict) and not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a Dataset object, the dataset is type of: " + str(type(dataset)))
        if type(dataset["train"]["image"][0]) != Image.Image and type(dataset["train"]["image"][0]) != PIL.PngImagePlugin.PngImageFile:
            raise ValueError("Dataset must contain images, but is type of: " + str(type(dataset["train"]["image"][0])))

        # Initalizing resulting dataset
        dataset_move_boxes_with_labels = None

        ## Convert dataset to list
        list_dataset = self.convert_dataset_to_list(dataset, "train", "image", "labels")

        try:
            ## Convert images (from RGB) to gray-scale
            list_of_gray_scaled = self.process_image_dataset_rgb_to_grayscale(list_dataset)

            ## Convert gray-scale images to binary images with otsu's method
            list_of_binary_images = self.process_image_dataset_gray_scaled_to_binary_with_threshold(list_of_gray_scaled)

            ## Generate image containing only grid lines
            list_of_only_grid_lines = self.process_image_dataset_binary_to_grid_lines(list_of_binary_images)

            ## Contour Algorithm
            column_for_contours = "list_of_contours"
            column_for_labels = "labels"
            list_of_contour_list_per_image = self.generate_image_dataset_binary_grid_to_list_of_contours(list_of_only_grid_lines, column_for_contours, column_for_labels)

            ## Cut out boxes with padding, give them a name and map them to the corresponding label
            column_for_contours = "list_of_contours"
            image_column = "image"
            label_column = "labels"
            dataset_move_boxes_with_labels = (
               self.generate_from_contour_list_and_image_list_cut_out_image_to_label_dataset(
                   list_of_contour_list_per_image, column_for_contours, list_of_gray_scaled, image_column, label_column))
        except ValueError as e:
            logger.error("ValueError: %s", e)
        except Exception as e:
            logger.error("Error: %s", e)

        return dataset_move_boxes_with_labels

    def convert_dataset_to_list(self, dataset: Dataset, split_column: str, feature_column: str, label_column: str) -> list:
        """
        Convert dataset of a defined split to list

        :param dataset:
        :return: list of split from dataset
        """

        # Initialize value to count how many images were not processed
        count_not_processed_images = 0

        # Select split of dataset
        dataset_of_split = dataset[split_column]

        # Resulting list
        res_list_of_dataset = []

        # List of dataset
        for i in range(0, len(dataset_of_split)):
            try:
                # Dictionary of feature and label
                dict_element = {feature_column: dataset_of_split[i][feature_column], label_column: dataset_of_split[i][label_column]}
                res_list_of_dataset.append(dict_element)
            except Exception as e:
                count_not_processed_images +=1
                raise Exception(f"The image with the index {i} could not be converted to list element "
                                f"will be assigned as \"image\":None for further processing.")

        count_of_correctly_processed_images = len(dataset_of_split)-count_not_processed_images
        logger.info(f"{count_of_correctly_processed_images} out of {len(dataset_of_split)} images in the dataset "
                f"were successfully processed into a list "
                f"and {count_not_processed_images} elements were excluded!")

        return res_list_of_dataset

    def process_image_dataset_rgb_to_grayscale(self, list_of_dataset: list) -> list:
        """
        Using the luminosity method: grayscale = 0.299 * R + 0.587 * G + 0.114 * B

        :param list_of_dataset: image dataset with images in RGB format and the corresponding text as target values
        :return: dataset with images in gray-scale format
        """

        # Initialize value to count how many images were not processed
        count_not_processed_images = 0

        for i in range(0, len(list_of_dataset)):
            list_of_dataset[i]["image"] = list_of_dataset[i]["image"].convert("L")

        count_of_correctly_processed_images = len(list_of_dataset)-count_not_processed_images
        logger.info(f"All images in the list were successfully processed into gray scale format!")

        return list_of_dataset

    def process_image_dataset_gray_scaled_to_binary_with_threshold(self, list_of_grayscaled_images: list) -> list:
        """
        Making the images in the dataset binary with Otsu's method

        :param list_of_grayscaled_images: image dataset with images in gray-scale format and the corresponding text as target values
        :return: dataset with images in binary format
        """
        # Initialize value to count how many images were not processed
        count_not_processed_images = 0

        # List of thresholds to calculate Average threshold
        thresholds = []

        for i in range(0, len(list_of_grayscaled_images)):
            try:
                threshold, list_of_grayscaled_images[i]["image"] = self.process_image_gray_scaled_to_binary_with_threshold(list_of_grayscaled_images[i]["image"])
                thresholds.append(threshold)
            except Exception as e:
                list_of_grayscaled_images[i]["image"] = None
                count_not_processed_images +=1
                logger.error(f"Error during processing image with index {i}, with error: {e}")

        # Calculate average threshold
        avg_threshold = round((sum(thresholds) / len(thresholds)), 2)

        # Calculate successfully processed images
        count_of_correctly_processed_images = len(list_of_grayscaled_images)-count_not_processed_images
        logger.info(f"{count_of_correctly_processed_images} out of {len(list_of_grayscaled_images)} images in the list "
                    f"were successfully processed into binary scale format with an average threshold of {avg_threshold} "
                    f"and {count_not_processed_images} where assigned with \"image\":None!")

        return list_of_grayscaled_images

    def process_image_gray_scaled_to_binary_with_threshold(self, image) -> tuple[float, Image]:
        """
        Uses Otsu's method to binarize the image.

        :param image: image in gray-scale format
        :return: image in binary format
        """
        if image is None:
            raise ValueError("Input image can not be None.")

        if not isinstance(image, Image.Image):
            raise ValueError(f"The input image must be of type PIL.Image, but is {type(image)}")

        np_image = np.array(image)
        count_of_dimensions = len(np_image.shape)
        if count_of_dimensions != 2:
            raise ValueError(f"Image is not in gray scale!")

        try:
            np_image = np.array(image)
            ret, thresh = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_image = Image.fromarray(thresh)

            return ret, binary_image

        except Exception as e:
            raise Exception(f"Error when converting to binary: {str(e)}")

    def process_image_dataset_binary_to_grid_lines(self, list_of_binary_images: list) -> list:
        """
        Using morphological operations to generate a binary image with only grid lines

        :param list_of_binary_images: image list with images in binary format and the corresponding text as target values
        :return: binary image with only grid lines
        """
        # Initialize value to count how many images were not processed
        count_not_processed_images = 0

        for i in range(0, len(list_of_binary_images)):
            try:
                list_of_binary_images[i]["image"] = self.process_binary_image_to_grid_lines(list_of_binary_images[i]["image"])
            except Exception as e:
                list_of_binary_images[i]["image"] = None
                count_not_processed_images +=1
                logger.error(f"Error during processing image with index {i} with error: {e}")

        # Calculate successfully processed images
        count_of_correctly_processed_images = len(list_of_binary_images)-count_not_processed_images
        logger.info(f"{count_of_correctly_processed_images} out of {len(list_of_binary_images)} images in the list "
                    f"were successfully processed into binary scale images with grid lines "
                    f"and {count_not_processed_images} where assigned with \"image\":None!")

        return list_of_binary_images

    def process_binary_image_to_grid_lines(self, image):
        """
        Using morphological operations to generate a binary image with only grid lines

        :param image: image in binary format
        :return: binary image with only grid lines
        """

        # Guard clauses
        if image is None:
            raise ValueError("Input image can not be None.")

        if not isinstance(image, Image.Image):
            raise ValueError(f"The input image must be of type PIL.Image, but is {type(image)}")

        np_img = np.array(image)
        is_image_binary = np.all(np.isin(np_img, [0, 255]))

        if not is_image_binary:
            raise ValueError(f"The input image must be in binary format!")

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

    def generate_image_dataset_binary_grid_to_list_of_contours(self, list_of_only_grid_lines: list, column_for_contours: str, column_for_labels: str) -> list:
        """
        Creates a list of contours per image, storing four point for the corners of each quadrilateral

        :param list_of_only_grid_lines: image dataset with images in binary format with grid lines and the corresponding text as target values
        :return: dataset with list of contours and the corresponding list of labels
        """
        # Initialize value to count how many images were not processed
        count_not_processed_images = 0

        # Initalizing resulting list
        len_of_list_of_only_grid_lines = len(list_of_only_grid_lines)
        res_contours_list = []
        for i in range(0, len_of_list_of_only_grid_lines):
            list_of_labels_for_image = list_of_only_grid_lines[i]["labels"]
            temp_contours = None
            dict_elem = {column_for_contours: temp_contours, column_for_labels: list_of_labels_for_image}
            res_contours_list.append(dict_elem)

            # Replace each image with the list of the contours for the image
        for i in range(0, len(list_of_only_grid_lines)):
            try:
                list_of_labels_for_image = list_of_only_grid_lines[i]["labels"]
                list_of_contours_for_image = self.generate_binary_grid_image_to_list_of_contours(list_of_only_grid_lines[i]["image"])
                dict_elem = {column_for_contours: list_of_contours_for_image, column_for_labels: list_of_labels_for_image}
                res_contours_list[i] = dict_elem
            except Exception as e:
                list_of_only_grid_lines[i]["image"] = None
                count_not_processed_images +=1
                logger.error(f"Error during processing image with index {i} with error: {e}")

            # Calculate successfully processed images
        count_of_correctly_processed_images = len(list_of_only_grid_lines)-count_not_processed_images
        logger.info(f"{count_of_correctly_processed_images} out of {len(list_of_only_grid_lines)} images in the list "
                    f"were successfully processed into a list of contours "
                    f"and {count_not_processed_images} where assigned with \"image\":None!")

        return res_contours_list

    def generate_binary_grid_image_to_list_of_contours(self, image):
        """
        Creates a list of contours for the image. The contours store four points for the corners of each quadrilateral

        :param image: image in binary format with grid lines
        :return: list of contours
        """

        # Guard clauses
        if image is None:
            raise ValueError("Input image can not be None.")

        if not isinstance(image, Image.Image):
            raise ValueError(f"The input image must be of type PIL.Image, but is {type(image)}")

        # Convert image to numpy
        np_img = np.array(image)
        is_image_binary = np.all(np.isin(np_img, [0, 255]))

        if not is_image_binary:
            raise ValueError(f"The input image must be in binary format!")

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
        median_area = float(np.median([cv2.contourArea(contour) for contour in simplified_contours]))
        filtered_contours = [contour for contour in simplified_contours if 0.5 * median_area <= cv2.contourArea(contour) <= 1.5 * median_area]

        if len(filtered_contours) == 120:
            # Sort contours based on their positions relative to one another
            sorted_contours = self.get_contour_precendence(filtered_contours)
            # Sort points in the 4 points of the corners of each move box contour
            sorted_contours_with_sorted_points = self.sort_points_in_list_of_contours(sorted_contours)
            return sorted_contours_with_sorted_points
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

    def sort_points_in_list_of_contours(self, list_of_contours: list) -> list:
        """
        Sort points in the 4 points of the corners of each move box contour. So the first point is the top left corner,
        the second point is the top right corner, the third point is the bottom right corner,
        and the fourth point is in the bottom left corner.
        Comment: And delete unnecessary dimension for point

        :param list_of_contours:
        :return:
        """
        res_list_of_contours = list_of_contours.copy()

        # Delete unnecessary dimension
        for index_contours, contour in enumerate(list_of_contours):
            res_list_of_contours[index_contours] = contour.squeeze()

        # Sort points in the 4 points of the corners of each move box contour
        for index_contours, contour in enumerate(res_list_of_contours):
            sorted_corners = []
            # Sorted for the top corners
            top_left_corner = sorted(contour, key=lambda point: point[1])
            # Select the two top corners
            top_corners = top_left_corner[0:2]
            # Sort the top corners based on their x-coordinate
            top_corners = sorted(top_corners, key=lambda point: point[0])
            # Add the top corners to the sorted_corners list
            sorted_corners.extend(top_corners)
            # Select the bottem corners
            bottom_corners = top_left_corner[2:4]
            # Sort the bottem corners based on their x-coordinate
            bottom_corners = sorted(bottom_corners, key=lambda point: point[0], reverse=True)
            # Add the bottem corners to the sorted_corners list
            sorted_corners.extend(bottom_corners)
            sorted_corners = np.array(sorted_corners)
            # Add the sorted_corners list to the res_list_of_contours list
            res_list_of_contours[index_contours] = sorted_corners

        return res_list_of_contours

    def generate_from_contour_list_and_image_list_cut_out_image_to_label_dataset(self,
                                                                                 list_of_contour_list_per_image: list,
                                                                                 contour_column: str,
                                                                                 list_of_gray_scaled: list,
                                                                                 image_column: str, label_column: str) -> Dataset:
        """


        :param list_of_contour_list_per_image:
        :param list_of_gray_scaled:
        :return:
        """

        if len(list_of_contour_list_per_image) != len(list_of_gray_scaled):
            raise ValueError(f"The two dataset are not the same size: "
                             f"Contour dataset: {len(list_of_contour_list_per_image)}, Image dataset: {len(list_of_gray_scaled)}")

        # Initialize value to count how many images were not processed
        count_not_processed_images = 0

        cut_out_image_to_label_list = []
        length_of_dataset = len(list_of_contour_list_per_image)

        for i in range(0, length_of_dataset):
            try:
                list_of_contours = list_of_contour_list_per_image[i][contour_column]
                image = list_of_gray_scaled[i][image_column]
                list_of_labels = list_of_gray_scaled[i][label_column]
                # Iterate through contour list and cut out images
                ## just cut out images where there is a label
                length_of_label_list = len(list_of_labels)
                for index_contours, contour in enumerate(list_of_contours[:length_of_label_list]):
                    cut_out_image = self.generate_from_four_contour_points_and_image_a_cut_out_image(list_of_contours, image)
                    label = list_of_labels[index_contours]
                    dict_element = {"label":label, "image":cut_out_image}
                    cut_out_image_to_label_list.append(dict_element)
            except Exception as e:
                count_not_processed_images +=1
                logger.error(f"Error during processing image with index {i} with error: {e}")

        # Define dataset object
        features = Features({
            "label": str,
            "image": datasets.Image()
        })
        # Convert list of dicts to dataset object
        dataset_cut_out_image_to_label_dataset = Dataset.from_list(cut_out_image_to_label_list, features=features)

        # Calculate successfully processed images
        count_of_correctly_processed_images = len(list_of_contour_list_per_image)-count_not_processed_images
        logger.info(f"{count_of_correctly_processed_images} out of {len(list_of_contour_list_per_image)} images in the list "
                    f"were successfully processed into a list of move boxes "
                    f"and {count_not_processed_images} where assigned with \"image\":None!")

        return dataset_cut_out_image_to_label_dataset

    def generate_from_four_contour_points_and_image_a_cut_out_image(self, list_contours, image: PIL.Image):
        """
        Cut out image with contours defined in list_contours from the bigger image

        :param list_contours:
        :param image:
        :return:
        """
        # Define corners
        top_left_corner = list_contours[0]
        bottom_right_corner = list_contours[2]

        # Define padding for cut out image
        percent_padding_on_top = 0.15
        percent_padding_on_bottom = 0.25
        padding_at_top = percent_padding_on_top*abs(bottom_right_corner[1]-top_left_corner[1])
        padding_at_bottom = percent_padding_on_bottom*abs(bottom_right_corner[1]-top_left_corner[1])

        # Redefine corners with padding
        top_left_corner[1] = top_left_corner[1] - padding_at_top
        bottom_right_corner[1] = bottom_right_corner[1] + padding_at_bottom

        cut_out_image = image.crop((top_left_corner[0], top_left_corner[1], bottom_right_corner[0], bottom_right_corner[1]))

        return cut_out_image