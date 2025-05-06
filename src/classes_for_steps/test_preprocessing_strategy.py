import unittest

import numpy as np
from datasets import Dataset

from src.classes_for_steps.ingest_data import HuggingFaceImageIngestor
from src.classes_for_steps.preprocessing_strategy import HuggingFacePreprocessingStrategy


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        dataset_name = "unprocessed_hcs"
        ingestor = HuggingFaceImageIngestor()
        cls.sut_preprocessing = HuggingFacePreprocessingStrategy()

        # When
        cls.dataset = ingestor.ingest_image_dataset_from_huggingface(owner=owner, dataset_name=dataset_name)

    def test_convert_dataset_to_list_Return_list_of_dataset_returns_list(self):
        # When
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
                           .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))

        # Then
        self.assertIsInstance(list_of_dataset, list)
        self.assertEqual(206, len(list_of_dataset))

    def test_process_image_dataset_rgb_to_grayscale_returns_list(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        # When
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)

        # Then
        self.assertIsInstance(grayscaled_list, list)
        for i in range(0, len(grayscaled_list)):
            sample_image = np.array(grayscaled_list[i]["image"])
            count_of_dimensions = len(sample_image.shape)
            self.assertEqual(2, count_of_dimensions)
        self.assertEqual(206, len(grayscaled_list))  # add assertion here

    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_returns_list(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)

        # When
        list_binary = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(grayscaled_list)

        # Then
        self.assertIsInstance(list_binary, list)
        for i in range(0, len(list_binary)):
            sample_image = np.array(list_binary[i]["image"])
            is_image_binary = np.all(np.isin(sample_image, [0, 255]))
            self.assertTrue(is_image_binary)
        self.assertEqual(206, len(list_binary))  # add assertion here

    def test_process_image_dataset_binary_to_grid_lines_returns_list(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)
        list_binary = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(grayscaled_list)

        # When
        list_of_binary_image_with_grid_lines = self.sut_preprocessing.process_image_dataset_binary_to_grid_lines(list_binary)

        # Then
        self.assertIsInstance(list_of_binary_image_with_grid_lines, list)
        for i in range(0, len(list_of_binary_image_with_grid_lines)):
            sample_image = np.array(list_of_binary_image_with_grid_lines[i]["image"])
            is_image_binary = np.all(np.isin(sample_image, [0, 255]))
            self.assertTrue(is_image_binary)
        self.assertEqual(206, len(list_of_binary_image_with_grid_lines))  # add assertion here

    def test_preprocess_image_dataset_return_dataset_with_cut_out_move_boxes(self):
        # When
        res_dataset = self.sut_preprocessing.preprocess_image_dataset(self.dataset)

        # Then
        self.assertIsInstance(res_dataset, Dataset)
        self.assertEqual(206, len(res_dataset["train"]))

if __name__ == '__main__':
    unittest.main()
