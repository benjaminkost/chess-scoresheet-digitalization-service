import logging
from abc import ABC, abstractmethod
from datasets import Dataset, Image
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ImageLabelDirToDatasetStrategy(ABC):
    @abstractmethod
    def get_dataset(self, path_to_image_dir: str, path_to_label_file: str) -> Dataset:
        """Abstract method to get a dataset from image and label files"""
        pass

class UnprocessedHcsImageLabelDirToDatasetStrategy(ImageLabelDirToDatasetStrategy):
    # method to create dict object of ground truth values for specific image
    # prerequisite: the moves in the ground truth values have to be in the correct move order
    def ground_truth_dict_for_image(self, image_name: str, abs_path_to_ground_truth_file: str):
        res = {}

        file_ground_truth = open(abs_path_to_ground_truth_file, "r")
        str_ground_truth = file_ground_truth.read()

        list_labels = str_ground_truth.split("\n")

        for label in list_labels:
            if image_name in label:
                # the location and the ground truth label are saved in a dict object
                res[label.split(" ")[0]] = label.split(" ")[1]

        return res

    # creates a dictionary inside a dictionary for main images and the sub image with the corresponding label
    def ground_truth_dict_for_all_images(self, path_to_image_dir: str, abs_path_to_ground_truth_file: str):
        res = {}

        list_images = os.listdir(path_to_image_dir)

        logging.info(f"Creating a dictionary inside a dictionary for main images and the sub image with the corresponding label."
                     f"With the path to the images: "
                     f"{path_to_image_dir} and the ground truth file: {abs_path_to_ground_truth_file}")

        for image_compl_name in list_images:
            if ".png" in image_compl_name:
                image_name = image_compl_name.split(".")[0]
                temp_dict = self.ground_truth_dict_for_image(image_name, abs_path_to_ground_truth_file)
                res[image_compl_name] = temp_dict

        return res

    def create_dataset_from_dict_with_sub_img(self, path_to_image_dir: str, img_label_dict):
        # List for Dataset
        data = []

        logging.info(f"Creating a dataset from a dictionary with the main image and the sub image with the corresponding label.")

        # Transform data
        for main_img, sub_images in img_label_dict.items():
            temp_labels = []
            image_path = os.path.join(path_to_image_dir, main_img)
            if os.path.exists(image_path):
                for sub_img, label in sub_images.items():
                    temp_labels.append(label)
                data.append({"image": image_path, "labels": temp_labels})

            dataset = Dataset.from_list(data).cast_column("image", Image())

        return dataset

    def get_dataset(self, path_to_image_dir: str, path_to_label_file: str) -> Dataset:
        ## Load Labels regarding there image names
        data_dict = self.ground_truth_dict_for_all_images(path_to_image_dir, path_to_label_file)

        ## Create dataset object from dict
        dataset = self.create_dataset_from_dict_with_sub_img(path_to_image_dir, data_dict)

        logging.info(f"Dataset created successfully.")

        return dataset

class ProcessedHcsImageLabelDirToDatasetStrategy(ImageLabelDirToDatasetStrategy):
    def ground_truth_dict_image_to_label(self, path_to_image_dir: str, abs_path_to_ground_truth_file: str):
        res = {}

        # all image names
        list_images = os.listdir(path_to_image_dir)
        # Sort them alphabetically
        list_images.sort()

        # ground truth values as string
        file_ground_truth = open(abs_path_to_ground_truth_file, "r")
        str_ground_truth = file_ground_truth.read()

        # ground truth values as list
        list_ground_truth = str_ground_truth.split("\n")
        # Sort them alphabetically
        list_ground_truth.sort()

        logging.info(f"Creating a dictionary inside a dictionary for main images and the sub image with the corresponding label."
                     f"With the path to the images: "
                     f"{path_to_image_dir} and the ground truth file: {abs_path_to_ground_truth_file}")

        # Create dict object with image_name corresponding to label
        for image_name in list_images:
            for ground_truth_value in list_ground_truth:
                if ground_truth_value.count(image_name) > 1:
                    ValueError(f"Error: For {image_name} are multiple labels in the ground truth file!")
                elif image_name in ground_truth_value:
                    res[image_name] = ground_truth_value.split(" ")[1]
                    break

        return res

    def create_dataset_from_dict_with_img_to_label(self, path_to_image_dir:str, img_label_dict):
        # List for Dataset
        data = []

        logging.info(f"Creating a dataset from a dictionary with the main image and the sub image with the corresponding label.")

        # Transform data
        for img, label in img_label_dict.items():
            image_path = os.path.join(path_to_image_dir, img)
            if os.path.exists(image_path):
                data.append({"image": image_path, "label": label})

            dataset = Dataset.from_list(data).cast_column("image", Image())

        return dataset

    def get_dataset(self, path_to_image_dir: str, path_to_label_file: str) -> Dataset:
        ## Make dict object with image and corresponding label
        processed_hcs_image_label_dict = self.ground_truth_dict_image_to_label(path_to_image_dir, path_to_label_file)

        ## Create dataset object from dict
        dataset_processed_hcs = self.create_dataset_from_dict_with_img_to_label(path_to_image_dir, processed_hcs_image_label_dict)

        logging.info(f"Dataset created successfully.")

        return dataset_processed_hcs

