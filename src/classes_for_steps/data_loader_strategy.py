from abc import ABC, abstractmethod
from datasets import Dataset

class DataLoaderStrategy(ABC):
    @abstractmethod
    def load_data(self, dataset: Dataset):
        pass

class HuggingFaceImageDatasetDataLoader(DataLoaderStrategy):
    def __init__(self, batch_size=50, data_set="train", batch_start=0):
        self._batch_size=batch_size
        self._data_set=data_set
        self._batch_start=batch_start

    def load_data(self, dataset: Dataset):
        res_list = []

        batch_start_index = self._batch_start
        batch_end_index = self._batch_start + self._batch_size + 1
        feature_column = dataset.features[0]
        label_column = dataset.features[1]
        for i in range(batch_start_index, dataset[self._data_set][batch_start_index:batch_end_index]):
            dict_element = {feature_column: dataset[i][feature_column], label_column: dataset[i][label_column]}
            res_list.append(dict_element)

        self._batch_start = batch_end_index

        return res_list