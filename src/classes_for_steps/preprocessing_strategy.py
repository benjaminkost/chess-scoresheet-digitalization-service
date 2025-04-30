from abc import ABC, abstractmethod
from PIL import Image
from datasets import Dataset
from pythreshold.global_th import otsu_threshold


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
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a Dataset object")
        if type(dataset["train"].features) != Image.Image:
            raise ValueError("Dataset must contain images")

        # Cut out move boxes
        processed_img_dataset = self.generate_move_boxes_from_image_dataset(dataset)

        # Map image to label
        res_dataset = self.map_move_boxes_to_label(processed_img_dataset)

        return res_dataset

    ##TODO: Implement Method
    def map_move_boxes_to_label(self, dataset: Dataset) -> Dataset:
        """Abstract method to map move boxes to a label"""
        return dataset

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

        return dataset

    ##TODO: Implement Method
    def generate_image_containing_only_grid_lines(self, image):
        """
        Use two long, thin kernels (one horizontal and one vertical) with sizes relative to input image dimensions,
        and morphological operations (erosion followed by dilation) with those kernels to generate an image containing
        only grid lines. - Digitization of Handwritten Chess Scoresheets with a BiLSTM Network

        :param image:
        :return:
        """

        res_img = None

        return res_img

    ##TODO: Implement Method
    def generate_list_of_image_with_grid_lines(self, image):
        res_img_list = None

        return res_img_list

    ##TODO: Implement Method
    def generate_contour_image(self, image):
        res_img = None

        return res_img

    ##TODO: Implement Method
    def process_image_dataset_with_grayscale(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {"image": x["image"].convert("L")})

        return dataset

    ##TODO: Implement Method
    def process_image_dataset_gray_scaled_to_binary_with_threshold(self, dataset: Dataset) -> Dataset:
        """
        The idea of this method cames from https://ieeexplore.ieee.org/document/4310076

        :param dataset:
        :return:
        """
        for img in dataset["train"]["image"]:
            otsu_threshold(img)

        return dataset