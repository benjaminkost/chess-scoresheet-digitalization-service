from torch import nn
from abc import ABC, abstractmethod


class Module(nn.Module, ABC):
    @abstractmethod
    def loss(self, y_hat, y):
        """

        :param y_hat: is the output of the forward function
        :param y: is the target
        :return:
        """
        pass

    @abstractmethod
    def forward(self, X):
        """

        :param X: features of the dataset
        :return:
        """
        pass

    @abstractmethod
    def plot(self, key, value, train):
        """

        :param key:
        :param value:
        :param train:
        :return:
        """
        pass

    @abstractmethod
    def training_step(self, batch):
        """

        :param batch: is a batch of the list of dicts with the images and the corresponding labels
        :return:
        """
        pass

    @abstractmethod
    def validation_step(self, batch):
        """

        :param batch: is a batch of the list of dicts with the images and the corresponding labels
        :return:
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """

        :return:
        """
        pass

