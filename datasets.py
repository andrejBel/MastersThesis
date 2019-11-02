from abc import ABC, abstractmethod

from tensorflow.keras import datasets
import numpy as np
import tensorflow.keras as keras

from global_functions import run_once
import constants

from typing import List, Tuple, Callable

class Dataset(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_number_of_classes(self) -> int:
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

    @abstractmethod
    @run_once
    def preprocess_images(self) -> None:
        pass

    @abstractmethod
    def get_train_images(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_train_labels(self) -> np.ndarray:
        pass

    def get_train_labels_one_hot(self) -> np.ndarray:
        return keras.utils.to_categorical(self.get_train_labels(), self.get_number_of_classes())

    @abstractmethod
    def get_test_images(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_test_labels(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_classes_names(self) -> List[str]:
        pass

    def get_train_label_class_name(self, picture_index) -> str:
      return self.get_classes_names()[self.get_train_labels()[picture_index]]

    def get_test_label_class_name(self, picture_index) ->str:
      return self.get_classes_names()[self.get_test_labels()[picture_index]]

    def get_test_labels_one_hot(self) -> np.ndarray:
        return keras.utils.to_categorical(self.get_test_labels(), self.get_number_of_classes())

    def get_input_shape(self) -> Tuple[int, int, int]:
        assert self.get_train_images().shape[1:] == self.get_test_images().shape[1:]
        return self.get_train_images().shape[1:]

    def process_image_for_plotting(self, image) -> np.ndarray:
        input_shape = self.get_input_shape()
        if input_shape[2] == 1:
            return image.reshape(input_shape[0], input_shape[1])
        else:
            return image

    

class KerasDataset(Dataset):

    def __init__(self, load_function):
        super().__init__()
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = load_function()
        self.preprocess_images()

    def preprocess_images(self) -> None:
        self.train_images = self.train_images.reshape(self.train_images.shape[0], self.train_images.shape[1], self.train_images.shape[2], -1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], self.test_images.shape[1], self.test_images.shape[2], -1)

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def get_train_images(self) -> np.ndarray:
        return self.train_images

    def get_train_labels(self) -> np.ndarray:
        return self.train_labels

    def get_test_images(self) -> np.ndarray:
        return self.test_images

    def get_test_labels(self) -> np.ndarray:
        return self.test_labels


class MnistDataset(KerasDataset):

    def __init__(self):
        super().__init__(datasets.mnist.load_data)

    def get_classes_names(self) -> List[str]:
        return ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
                'Nine']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self) -> str:
        return constants.Datasets.MNIST

class FashionMnistDataset(KerasDataset):

    def __init__(self):
        super().__init__(datasets.fashion_mnist.load_data)

    def get_classes_names(self) -> List[str]:
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self) -> str:
        return constants.Datasets.FASHION_MNIST


class Cifar10Dataset(KerasDataset):

    def __init__(self):
        super().__init__(datasets.cifar10.load_data)

    def get_classes_names(self) -> List[str]:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self) -> str:
        return constants.Datasets.CIFAR_10



class DatasetFactory:

    def __init__(self):
        self.dataset_makers = {}
        self.register_dataset(constants.Datasets.MNIST, MnistDataset)
        self.register_dataset(constants.Datasets.FASHION_MNIST, FashionMnistDataset)
        self.register_dataset(constants.Datasets.CIFAR_10, Cifar10Dataset)

    def register_dataset(self, name : str, maker):
        self.dataset_makers[name] = maker

    def make_dataset(self, type) -> Dataset:
        maker = self.dataset_makers.get(type, None)
        if not maker:
            raise ValueError(type)
        return maker()


DatasetProvider = Callable[[], Dataset]

class DatasetProviderClass():

    def __init__(self, provider: DatasetProvider):
        self.provider: DatasetProvider = provider

    def __call__(self) -> Dataset:
        return self.provider()

