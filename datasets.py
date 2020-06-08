from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets

import constants
from global_functions import run_once


class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


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

    def get_test_label_class_name(self, picture_index) -> str:
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

    def provide(self, **ignore):
        return self


class KerasDataset(Dataset, Singleton):

    def __init__(self, load_function):
        super().__init__()
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = load_function()
        self.preprocess_images()
        self.preprocess_labels()
        self.filter()
        self.train_images.setflags(write=False)
        self.train_labels.setflags(write=False)
        self.test_images.setflags(write=False)
        self.test_labels.setflags(write=False)

    def filter(self):
        pass

    def preprocess_images(self) -> None:
        self.train_images = self.train_images.reshape(self.train_images.shape[0], self.train_images.shape[1],
                                                      self.train_images.shape[2], -1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], self.test_images.shape[1],
                                                    self.test_images.shape[2], -1)

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def preprocess_labels(self) -> None:
        pass

    def get_train_images(self) -> np.ndarray:
        return self.train_images

    def get_train_labels(self) -> np.ndarray:
        return self.train_labels

    def get_test_images(self) -> np.ndarray:
        return self.test_images

    def get_test_labels(self) -> np.ndarray:
        return self.test_labels


class FilterableKerasDataset(KerasDataset):

    @abstractmethod
    def get_chosen_classes(self) -> List[int]:
        pass

    def filter(self):
        chosen_classes = self.get_chosen_classes()

        indexes_train_labels = np.array([index for index, x in enumerate(self.train_labels) if x in chosen_classes])
        self.train_labels = self.train_labels[indexes_train_labels]
        self.train_images = self.train_images[indexes_train_labels]

        indexes_test_labels = np.array([index for index, x in enumerate(self.test_labels) if x in chosen_classes])
        self.test_labels = self.test_labels[indexes_test_labels]
        self.test_images = self.test_images[indexes_test_labels]


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


class FashionMnistDatasetFiveClasses(FilterableKerasDataset):

    def __init__(self):
        super().__init__(datasets.fashion_mnist.load_data)

    def get_chosen_classes(self) -> List[int]:
        return [0, 1, 2, 3, 4]

    def get_classes_names(self) -> List[str]:
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat']

    def get_number_of_classes(self) -> int:
        return 5

    def get_missing_train_images(self):
        inchosen_classes = [5, 6, 7, 8, 9]
        indexes_train_labels = np.array(
            [index for index, x in enumerate(FashionMnistDataset().get_train_labels()) if x in inchosen_classes])
        original_train_images = FashionMnistDataset().get_train_images()
        return original_train_images[indexes_train_labels]

    def get_missing_test_images(self):
        inchosen_classes = [5, 6, 7, 8, 9]
        indexes_test_labels = np.array(
            [index for index, x in enumerate(FashionMnistDataset().get_test_labels()) if x in inchosen_classes])
        original_test_images = FashionMnistDataset().get_test_images()
        return original_test_images[indexes_test_labels]

    def get_dataset_name(self) -> str:
        return constants.Datasets.FASHION_MNIST_FIVE


class FashionMnistDatasetOtherFiveClasses(FilterableKerasDataset):

    def __init__(self):
        super().__init__(datasets.fashion_mnist.load_data)

    def get_chosen_classes(self) -> List[int]:
        return [5, 6, 7, 8, 9]

    def get_classes_names(self) -> List[str]:
        return ['Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def get_number_of_classes(self) -> int:
        return 5

    def get_dataset_name(self) -> str:
        return constants.Datasets.FASHION_MNIST_FIVE_OTHER


class Cifar10PreprocessLabelsBase(KerasDataset):

    def preprocess_labels(self) -> None:
        self.train_labels = np.array([label[0] for label in self.train_labels])
        self.test_labels = np.array([label[0] for label in self.test_labels])


class Cifar10Dataset(Cifar10PreprocessLabelsBase):

    def __init__(self):
        super().__init__(datasets.cifar10.load_data)

    # def preprocess_images(self) -> None:
    #     # super().preprocess_images()
    #     self.train_images = self.train_images.astype('float32')
    #     self.test_images = self.test_images.astype('float32')
    #
    #     # z score
    #     # mean_train = np.mean(self.train_images, axis=(0, 1, 2))
    #     # std_train = np.std(self.train_images, axis=(0, 1, 2))
    #     # self.train_images = (self.train_images - mean_train) / (std_train + 1e-7)
    #     # self.test_images = (self.test_images - mean_train) / (std_train + 1e-7)
    #
    #     self.train_images = self.train_images / 255.0
    #     self.test_images = self.test_images / 255.0
    #
    #     self.train_images = self.train_images.reshape(self.train_images.shape[0], self.train_images.shape[1],
    #                                                   self.train_images.shape[2], -1)
    #     self.test_images = self.test_images.reshape(self.test_images.shape[0], self.test_images.shape[1],
    #                                                 self.test_images.shape[2], -1)

    def get_classes_names(self) -> List[str]:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self) -> str:
        return constants.Datasets.CIFAR_10


class CifarGrayScaledBase(Cifar10PreprocessLabelsBase):

    def preprocess_images(self) -> None:
        import cv2
        self.train_images = self.train_images.astype('float32')
        self.train_images = np.array(
            [cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_LINEAR) for image in self.train_images])
        self.train_images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in self.train_images])

        self.test_images = self.test_images.astype('float32')
        self.test_images = np.array(
            [cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_LINEAR) for image in self.test_images])
        self.test_images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in self.test_images])

        # # z score
        # mean_train = np.mean(self.train_images, axis=(0, 1, 2))
        # std_train = np.std(self.train_images, axis=(0, 1, 2))
        # self.train_images = (self.train_images - mean_train) / (std_train + 1e-7)
        # self.test_images = (self.test_images - mean_train) / (std_train + 1e-7)

        # spravi spravy dimension a podeli da na interval 0 .. 1
        super().preprocess_images()

        # self.train_images = self.train_images / 255.0
        # self.test_images = self.test_images / 255.0
        #
        # self.train_images = self.train_images.reshape(self.train_images.shape[0], self.train_images.shape[1],
        #                                               self.train_images.shape[2], -1)
        # self.test_images = self.test_images.reshape(self.test_images.shape[0], self.test_images.shape[1],
        #                                             self.test_images.shape[2], -1)


class Cifar10GrayDataset(CifarGrayScaledBase):

    def __init__(self):
        super().__init__(datasets.cifar10.load_data)

    def get_classes_names(self) -> List[str]:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self) -> str:
        return constants.Datasets.CIFAR_10 + ' gray'


class Cifar10GrayFiveDataset(FilterableKerasDataset, CifarGrayScaledBase):

    def __init__(self):
        super().__init__(datasets.cifar10.load_data)

    def get_chosen_classes(self) -> List[int]:
        return [0, 1, 2, 3, 4]

    def get_classes_names(self) -> List[str]:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer']

    def get_number_of_classes(self) -> int:
        return 5

    def get_dataset_name(self) -> str:
        return constants.Datasets.CIFAR_10_FIVE


class Cifar10GrayFiveOtherDataset(FilterableKerasDataset, CifarGrayScaledBase):

    def __init__(self):
        super().__init__(datasets.cifar10.load_data)

    def get_chosen_classes(self) -> List[int]:
        return [5, 6, 7, 8, 9]

    def get_classes_names(self) -> List[str]:
        return ['dog', 'frog', 'horse', 'ship', 'truck']

    def get_number_of_classes(self) -> int:
        return 5

    def get_dataset_name(self) -> str:
        return constants.Datasets.CIFAR_10_FIVE_OTHER


import models


class DerivatedDataset(Dataset):

    def __init__(self, original_dataset: Dataset, models: 'models.Models'):
        super().__init__()
        self.original_dataset: Dataset = original_dataset
        self.models = models
        self.train_images = self.original_dataset.get_train_images()
        self.test_images = self.original_dataset.get_test_images()
        self.preprocess_images()
        self.train_images.setflags(write=False)
        self.test_images.setflags(write=False)

    def get_number_of_classes(self) -> int:
        return self.original_dataset.get_number_of_classes()

    def get_dataset_name(self) -> str:
        return self.original_dataset.get_dataset_name() + "_" + constants.Datasets.DERIVATED

    def preprocess_images(self) -> None:
        self.train_images = self.models.autoencoder.predict(self.train_images)
        self.test_images = self.models.autoencoder.predict(self.test_images)

    def get_train_images(self) -> np.ndarray:
        return self.train_images

    def get_train_labels(self) -> np.ndarray:
        return self.original_dataset.get_train_labels()

    def get_test_images(self) -> np.ndarray:
        return self.test_images

    def get_test_labels(self) -> np.ndarray:
        return self.original_dataset.get_test_labels()

    def get_classes_names(self) -> List[str]:
        return self.original_dataset.get_classes_names()


class MergedDataset(Dataset):

    def __init__(self, original_dataset: Dataset, derivated_dataset: DerivatedDataset):
        super().__init__()
        self.original_dataset: Dataset = original_dataset
        self.derivated_dataset = derivated_dataset
        if not hasattr(derivated_dataset,
                       "original_dataset") or self.original_dataset != derivated_dataset.original_dataset:
            raise Exception("Wrong parameters")
        self.train_images = np.concatenate(
            (self.original_dataset.get_train_images(), self.derivated_dataset.get_train_images())
            , axis=0)
        self.train_labels = np.concatenate(
            (self.original_dataset.get_train_labels(), self.derivated_dataset.get_train_labels())
            , axis=0)
        self.test_images = np.concatenate(
            (self.original_dataset.get_test_images(), self.derivated_dataset.get_test_images())
            , axis=0)
        self.test_labels = np.concatenate(
            (self.original_dataset.get_test_labels(), self.derivated_dataset.get_test_labels())
            , axis=0)

        self.train_images.setflags(write=False)
        self.train_labels.setflags(write=False)
        self.test_images.setflags(write=False)
        self.test_labels.setflags(write=False)

    def get_number_of_classes(self) -> int:
        return self.original_dataset.get_number_of_classes()

    def get_dataset_name(self) -> str:
        return self.original_dataset.get_dataset_name() + "_" + constants.Datasets.MERGED

    def preprocess_images(self) -> None:
        pass

    def get_train_images(self) -> np.ndarray:
        return self.train_images

    def get_train_labels(self) -> np.ndarray:
        return self.train_labels

    def get_test_images(self) -> np.ndarray:
        return self.test_images

    def get_test_labels(self) -> np.ndarray:
        return self.test_labels

    def get_classes_names(self) -> List[str]:
        return self.original_dataset.get_classes_names()


DatasetProvider = Callable[[], Dataset]


class DatasetProviderClass():

    def __init__(self, provider: DatasetProvider):
        self.provider: DatasetProvider = provider

    def __call__(self) -> Dataset:
        return self.provider()


if __name__ == "__main__":
    import numpy as np

    c = Cifar10GrayFiveOtherDataset()
    m = np.mean(c.get_train_images())
    print(m)
