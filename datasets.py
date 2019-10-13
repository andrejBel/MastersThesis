from abc import ABC, abstractmethod

from tensorflow import keras

from global_functions import run_once
from constants import Constants


class Dataset(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_number_of_classes(self) -> int:
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

    @abstractmethod
    @run_once
    def preprocess_images(self):
        pass

    @abstractmethod
    def get_train_images(self):
        pass

    @abstractmethod
    def get_train_labels(self):
        pass

    def get_train_labels_one_hot(self):
        keras.utils.to_categorical(self.get_train_labels(), self.get_number_of_classes())

    @abstractmethod
    def get_test_images(self):
        pass

    @abstractmethod
    def get_test_labels(self):
        pass

    @abstractmethod
    def get_classes_names(self):
        pass

    def get_train_label_class_name(self, picture_index):
      return self.get_classes_names()[self.get_train_labels()[picture_index]]

    def get_test_label_class_name(self, picture_index):
      return self.get_classes_names()[self.get_test_labels()[picture_index]]

    def get_test_labels_one_hot(self):
        keras.utils.to_categorical(self.get_test_labels(), Dataset.get_number_of_classes())

    def get_input_shape(self):
        assert self.get_train_images().shape[1:] == self.get_test_images().shape[1:]
        return self.get_train_images().shape[1:]

    def process_image_for_plotting(self, image):
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

    def preprocess_images(self):
        self.train_images = self.train_images.reshape(self.train_images.shape[0], self.train_images.shape[1], self.train_images.shape[2], -1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], self.test_images.shape[1], self.test_images.shape[2], -1)

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def get_train_images(self):
        return self.train_images

    def get_train_labels(self):
        return self.train_labels

    def get_test_images(self):
        return self.test_images

    def get_test_labels(self):
        return self.test_labels


class MnistDataset(KerasDataset):

    def __init__(self):
        super().__init__(keras.datasets.mnist.load_data)

    def get_classes_names(self):
        return ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
                'Nine']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self):
        return Constants.Datasets.MNIST.name()


class FashionMnistDataset(KerasDataset):

    def __init__(self):
        super().__init__(keras.datasets.fashion_mnist.load_data)

    def get_classes_names(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self):
        return Constants.Datasets.FASHION_MNIST.name()


class Cifar10Dataset(KerasDataset):

    def __init__(self):
        super().__init__(keras.datasets.cifar10.load_data)

    def get_classes_names(self):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def get_number_of_classes(self) -> int:
        return 10

    def get_dataset_name(self):
        return Constants.Datasets.CIFAR_10.name()


class DatasetFactory:

    def __init__(self):
        self.dataset_makers = {}
        self.register_dataset(Constants.Datasets.MNIST, MnistDataset)
        self.register_dataset(Constants.Datasets.FASHION_MNIST, FashionMnistDataset)
        self.register_dataset(Constants.Datasets.CIFAR_10, Cifar10Dataset)

    def register_dataset(self, type, maker):
        self.dataset_makers[type] = maker

    def make_dataset(self, type) -> Dataset:
        maker = self.dataset_makers.get(type, None)
        if not maker:
            raise ValueError(type)
        return maker()

