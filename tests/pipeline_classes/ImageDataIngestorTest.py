import unittest

from src.pipeline_classes.impl.ImageDataIngestorImpl import ImageDataIngestorImpl


class ImageDataIngestorTest(unittest.TestCase):

    def test_ingest_image_dataset_from_huggingface_Chesscorner_HCS_Dataset_csv_ReturnsDS(self):
        # Give
        owner = "Chesscorner"
        dataset_name = "HCS_Dataset-csv"
        ingestor = ImageDataIngestorImpl()

        # When
        dataset = ingestor.ingest_image_dataset_from_huggingface(owner=owner, dataset_name=dataset_name)

        # Then
        self.assertEqual(206, len(dataset["train"]))  # add assertion here


if __name__ == '__main__':
    unittest.main()
