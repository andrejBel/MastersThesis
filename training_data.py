from typing import Callable, Optional, Union
from global_functions import auto_str_repr
from abc import ABC, abstractmethod
from datasets import Dataset
import numpy as np
import constants

TrainingDataProvider = Callable[..., Union['BasicTrainingData', 'TrainingDataAutoencoderClassifier']]


class SaveInfo():  # TODO throug inheritance
    pass


@auto_str_repr
class TrainParameters():

    def __init__(self, epochs: int, batch_size: int, patience: int, validate: bool, train_data_rate: float,
                 min_delta: float) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validate = validate
        self.train_data_rate = train_data_rate
        self.min_delta = min_delta


@auto_str_repr
class BasicTrainParametersAutoencoder(TrainParameters):

    def __init__(self, epochs: int, batch_size: int, patience: int, validate: bool, train_data_rate: float,
                 min_delta: float, log_path: Optional[str], save_weights: Optional[bool]):
        super().__init__(epochs, batch_size, patience, validate, train_data_rate, min_delta)
        self.log_path = log_path
        self.save_weights = save_weights


@auto_str_repr
class BasicTrainParametersClassifier(TrainParameters):

    def __init__(self, epochs: int, batch_size: int, patience: int, validate: bool, train_data_rate: float,
                 min_delta: float, autoencoder_layers_trainable_during_classifier_training: bool,
                 log_path: Optional[str], save_weights: Optional[bool]):
        super().__init__(epochs, batch_size, patience, validate, train_data_rate, min_delta)
        self.autoencoder_layers_trainable_during_classifier_training = autoencoder_layers_trainable_during_classifier_training
        self.log_path = log_path
        self.save_weights = save_weights


@auto_str_repr
class BasicTrainParametersTwoModels():

    def __init__(self, train_parameters_autoencoder: TrainParameters, train_parameters_classifier: TrainParameters,
                 training_cycles: int,
                 autoencoder_layers_trainable_during_classifier_training: bool,
                 log_path: Optional[str], save_weights: Optional[bool]):
        self.train_parameters_autoencoder = train_parameters_autoencoder
        self.train_parameters_classifier = train_parameters_classifier
        self.training_cycles = training_cycles
        self.autoencoder_layers_trainable_during_classifier_training = autoencoder_layers_trainable_during_classifier_training
        self.log_path = log_path
        self.save_weights = save_weights


@auto_str_repr
class BasicTrainParametersAutoClassifier(TrainParameters):

    def __init__(self, epochs: int, batch_size: int, patience: int, validate: bool, train_data_rate: float,
                 min_delta: float, loss_weigh_decoder: Optional[float], loss_weight_classifier: Optional[float],
                 log_path: Optional[str], save_weights: Optional[bool]):
        super().__init__(epochs, batch_size, patience, validate, train_data_rate, min_delta)
        self.loss_weigh_decoder = loss_weigh_decoder
        self.loss_weight_classifier = loss_weight_classifier
        self.log_path = log_path
        self.save_weights = save_weights


@auto_str_repr
class BasicTrainingData:

    def __init__(self, x, y, validation_data):
        self.x = x
        self.y = y
        self.validation_data = validation_data


class BasicTrainingDataGeneratorAutoencoder():

    def __init__(self, use_random_batch=True, random_start_index=None) -> None:
        self.use_random_batch = use_random_batch
        self.random_start_index = None

    def __call__(self, dataset: Dataset, train_data_rate, *args, **kwargs) -> BasicTrainingData:
        train_images = dataset.get_train_images()
        train_dataset_lenght = len(train_images)
        number_of_trainig_images = train_dataset_lenght * train_data_rate
        possible_batches = train_dataset_lenght / number_of_trainig_images
        if self.random_start_index is None:
            self.random_start_index = np.random.randint(0, possible_batches) if self.use_random_batch else 0
        start_batch_index = int(self.random_start_index * number_of_trainig_images)
        end_batch_index = int(start_batch_index + number_of_trainig_images)
        print('start_batch_index:', start_batch_index, ',end_batch_index:', end_batch_index)
        train_images = train_images[start_batch_index: end_batch_index]
        test_images = dataset.get_test_images()
        return BasicTrainingData(x=train_images, y=train_images,
                                 validation_data=(test_images, test_images))


class BasicTrainingDataGeneratorClassifier():

    def __init__(self, use_random_batch=True, random_start_index=None) -> None:
        self.use_random_batch = use_random_batch
        self.random_start_index = random_start_index

    def __call__(self, dataset: Dataset, train_data_rate, *args, **kwargs) -> BasicTrainingData:
        train_images = dataset.get_train_images()
        train_labels_one_hot = dataset.get_train_labels_one_hot()
        assert len(train_images) == len(train_labels_one_hot)
        train_dataset_lenght = len(train_images)
        number_of_trainig_images = train_dataset_lenght * train_data_rate
        possible_batches = train_dataset_lenght / number_of_trainig_images
        if self.random_start_index is None:
            self.random_start_index = np.random.randint(0, possible_batches) if self.use_random_batch else 0
        start_batch_index = int(self.random_start_index * number_of_trainig_images)
        end_batch_index = int(start_batch_index + number_of_trainig_images)
        print('start_batch_index:', start_batch_index, ',end_batch_index:', end_batch_index)
        train_images = train_images[start_batch_index: end_batch_index]
        train_labels_one_hot = train_labels_one_hot[start_batch_index: end_batch_index]
        test_images = dataset.get_test_images()
        test_labels = dataset.get_test_labels()
        test_labels_one_hot = dataset.get_test_labels_one_hot()
        return BasicTrainingData(x=train_images, y=train_labels_one_hot,
                                 validation_data=(test_images, test_labels_one_hot))


@auto_str_repr
class TrainingDataAutoencoderClassifier:

    def __init__(self, training_data_autoencoder: BasicTrainingData, training_data_classifier: BasicTrainingData):
        self.training_data_autoencoder = training_data_autoencoder
        self.training_data_classifier = training_data_classifier


class BasicTrainingDataGeneratorAutoencoderClassifier():

    def __init__(self, use_random_batch=True) -> None:
        self.use_random_batch = use_random_batch
        self.random_start_index_autoencoder = None
        self.random_start_index_classifier = None

    def __call__(self, dataset: Dataset, train_data_rate_autoencoder, train_data_rate_classifier, *args, **kwargs):
        train_images = dataset.get_train_images()
        train_dataset_lenght = len(train_images)

        number_of_training_images_autoencoder = train_dataset_lenght * train_data_rate_autoencoder
        possible_batches_autoencoder = train_dataset_lenght / number_of_training_images_autoencoder
        if self.random_start_index_autoencoder is None:
            self.random_start_index_autoencoder = np.random.randint(0,
                                                                    possible_batches_autoencoder) if self.use_random_batch else 0

        number_of_training_images_classifier = train_dataset_lenght * train_data_rate_classifier
        possible_batches_classifier = train_dataset_lenght / number_of_training_images_classifier
        if self.random_start_index_classifier is None:
            self.random_start_index_classifier = np.random.randint(0,
                                                                   possible_batches_classifier) if self.use_random_batch else 0

        return TrainingDataAutoencoderClassifier(
            BasicTrainingDataGeneratorAutoencoder(self.use_random_batch, self.random_start_index_autoencoder)
            (dataset, train_data_rate_autoencoder),
            BasicTrainingDataGeneratorClassifier(self.use_random_batch, self.random_start_index_classifier)
            (dataset, train_data_rate_classifier)
        )


class BasicTrainingDataGeneratorAutoClassifier():

    def __init__(self, use_random_batch=True, random_start_index=None) -> None:
        self.use_random_batch = use_random_batch
        self.random_start_index = random_start_index

    def __call__(self, dataset: Dataset, train_data_rate, *args, **kwargs) -> BasicTrainingData:
        train_images = dataset.get_train_images()
        train_labels_one_hot = dataset.get_train_labels_one_hot()
        assert len(train_images) == len(train_labels_one_hot)
        train_dataset_lenght = len(train_images)
        number_of_trainig_images = train_dataset_lenght * train_data_rate
        possible_batches = train_dataset_lenght / number_of_trainig_images
        if self.random_start_index is None:
            self.random_start_index = np.random.randint(0, possible_batches) if self.use_random_batch else 0

        start_batch_index = int(self.random_start_index * number_of_trainig_images)
        end_batch_index = int(start_batch_index + number_of_trainig_images)
        print('start_batch_index:', start_batch_index, ',end_batch_index:', end_batch_index)
        train_images = train_images[start_batch_index: end_batch_index]
        train_labels_one_hot = train_labels_one_hot[start_batch_index: end_batch_index]
        test_images = dataset.get_test_images()
        test_labels_one_hot = dataset.get_test_labels_one_hot()
        return BasicTrainingData(x=train_images, y={constants.Models.DECODED_OUT: train_images,
                                                    constants.Models.CLASSIFIER_OUT: train_labels_one_hot},
                                 validation_data=(test_images, {constants.Models.DECODED_OUT: test_images,
                                                                constants.Models.CLASSIFIER_OUT: test_labels_one_hot}))


PossibleTrainingparameters = Union[
    BasicTrainParametersAutoencoder, BasicTrainParametersClassifier, BasicTrainParametersTwoModels, BasicTrainParametersAutoClassifier]
