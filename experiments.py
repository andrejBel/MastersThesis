from abc import ABC, abstractmethod
from tabnanny import verbose

from tensorflow import keras

from keras.models import Model
from global_functions import coalesce
import constants
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from datasets import Dataset, DatasetProvider
from pathlib import Path
import collections

from global_functions import auto_str_repr
from models import Models, ModelProviderBase, ExistingModelProvider
from datetime import datetime
import jsonpickle
import numpy as np

from copy import deepcopy
from training_data import TrainParameters, BasicTrainParametersAutoencoder, BasicTrainingData, TrainingDataProvider, \
    BasicTrainParametersTwoModels, TrainingDataAutoencoderClassifier, PossibleTrainingparameters, \
    BasicTrainParametersClassifier, BasicTrainParametersAutoClassifier, BasicTrainingDataGeneratorAutoencoder, \
    BasicTrainingDataGeneratorAutoClassifier, BasicTrainingDataGeneratorClassifier

from callbacks import CustomEarlyStopping


@auto_str_repr
class TrainHistoryModel:
    def __init__(self, model_name: str, value_to_monitor: str, stopped_epoch: Optional[int], best_epoch: Optional[int],
                 monitor_best_value: Optional[float]):
        self.model_name = model_name
        self.stopped_epoch = stopped_epoch
        self.best_epoch = best_epoch
        self.value_to_monitor = value_to_monitor
        self.monitor_best_value = monitor_best_value
        self.metrics: Set[str] = set()
        self.values = collections.defaultdict(list)

    def set_stopped_epoch(self, stopped_epoch):
        self.stopped_epoch = stopped_epoch

    def set_best_epoch(self, best_epoch):
        self.best_epoch = best_epoch

    def set_monitor_best_value(self, monitor_best_value):
        self.monitor_best_value = monitor_best_value

    @staticmethod
    def init_from_trained_model(model_name: str, value_to_monitor: str, stopped_epoch: int, best_epoch: int,
                                monitor_best_value: float):
        return TrainHistoryModel(model_name, value_to_monitor, stopped_epoch, best_epoch, monitor_best_value)

    @staticmethod
    def init_before_training(model_name: str, value_to_monitor: str):
        return TrainHistoryModel(model_name, value_to_monitor, None, None, None)

    def add_metric_value(self, metric: str, value):
        self.metrics.add(metric)
        self.values[metric].extend(list(value))

    def add_history_dict(self, history: Dict):
        for metric, value in history.items():
            self.add_metric_value(metric, value)


@auto_str_repr
class TrainHistory:

    def __init__(self, train_parameters: PossibleTrainingparameters, train_history_model_list: List[TrainHistoryModel],
                 experiment_type: str, dataset_name: str):
        self.train_parameters = train_parameters
        self.train_history_model_list = train_history_model_list
        self.experiment_type = experiment_type
        self.dataset_name = dataset_name


@auto_str_repr
class ModelInfo():

    def __init__(self, path_to_model: Optional[str]):
        self.path_to_model = path_to_model


@auto_str_repr
class ExperimentResult:

    def __init__(self, experiment: 'ExperimentBase',
                 train_history: TrainHistory, model_infos: List[ModelInfo]):
        self.experiment = deepcopy(experiment)
        self.class_name = experiment.__class__.__name__
        self.train_history = train_history
        self.model_infos = model_infos


class ExperimentBase(ABC):

    def __init__(self, dataset_provider: DatasetProvider, model_provider: ModelProviderBase,
                 train_data_provider: TrainingDataProvider):
        self.dataset_provider: DatasetProvider = dataset_provider
        self.model_provider: ModelProviderBase = model_provider
        self.train_data_provider = train_data_provider

    @abstractmethod
    def get_experiment_type(self) -> str:
        pass

    @abstractmethod
    def train(self, parameters: TrainParameters):
        pass

    @staticmethod
    def get_model_path(dataset_name: str, model_name: str):
        now = datetime.now()
        return constants.Paths.OUTPUT_DIRECTORY + '_' + model_name + '_' + dataset_name + now.strftime(
            "_%d_%m_%Y_%H_%M_%S") + '.h5'

    @staticmethod
    def load_experiment_results(log_file: str, load_whole_trained_model: bool = True) -> Optional[
        List[ExperimentResult]]:
        path_to_log_file = Path(log_file)
        if Path(path_to_log_file).exists():
            text = Path(log_file).read_text()
            experiment_result_list: List[ExperimentResult] = jsonpickle.decode(text)

            if load_whole_trained_model:
                for experimentResult in experiment_result_list:
                    paths_to_models = [model_info.path_to_model for model_info in experimentResult.model_infos]
                    if any(path is None for path in paths_to_models):
                        continue

                    def provide_existing_model(**ignore) -> Models:
                        model_list = [keras.models.load_model(path_to_model) for path_to_model in paths_to_models]
                        models: Models = Models(*model_list)
                        return models

                    experimentResult.experiment.model_provider = provide_existing_model

            return experiment_result_list
        else:
            return None

    @staticmethod
    def provide_existing_model_from_log(log_file, predicate_for_chosen_model_provider) -> ExistingModelProvider:
        experiments_results: List[ExperimentResult] = ExperimentBase.load_experiment_results(log_file=log_file,
                                                                                             load_whole_trained_model=True)
        experiment_result: ExperimentResult = predicate_for_chosen_model_provider(experiments_results)
        return ExistingModelProvider(experiment_result.experiment.model_provider)

    @staticmethod
    def create_predicate_for_choosing_best_model(model_name: str, metric: str):
        def get_metric_value(result: ExperimentResult):
            model = next(
                (model for model in result.train_history.train_history_model_list if model.model_name == model_name),
                None)
            assert metric == model.value_to_monitor
            return model.monitor_best_value

        return lambda experiments_results: max(experiments_results, key=get_metric_value)

    @staticmethod
    def save_model_paths(dataset_name: str, models: Models) -> List[ModelInfo]:
        model_info: List[ModelInfo] = []
        for model in models:
            model_path = ExperimentBase.get_model_path(dataset_name, model.name)
            model.save(model_path)
            model_info.append(ModelInfo(model_path))
        return model_info

    @staticmethod
    def get_empty_model_paths(models: Models) -> List[ModelInfo]:
        return [ModelInfo(None) for model in models]

    @staticmethod
    def save_experiment_result_to_file(log_path: str, result: ExperimentResult):
        previous_logs: List[ExperimentResult] = coalesce(
            ExperimentBase.load_experiment_results(log_path, False), [])
        previous_logs.append(result)
        encoded = jsonpickle.encode(previous_logs)
        Path(log_path).write_text(encoded)

    @staticmethod
    def create_train_history_model_from_training(history, early_stopping: CustomEarlyStopping,
                                                 parameters: TrainParameters) -> TrainHistoryModel:
        stopped_epoch = early_stopping.last_epoch + 1
        best_epoch = early_stopping.best_epoch + 1
        value_to_monitor = early_stopping.monitor
        monitor_best_value = early_stopping.best

        train_history: TrainHistoryModel = TrainHistoryModel.init_from_trained_model(history.model.name,
                                                                                     value_to_monitor,
                                                                                     stopped_epoch,
                                                                                     best_epoch,
                                                                                     monitor_best_value)
        train_history.add_history_dict(history.history)
        return train_history

    def process_experiment_training(self, dataset: Dataset, parameters: PossibleTrainingparameters, models: Models,
                                    train_history_model_list: List[TrainHistoryModel]) -> ExperimentResult:
        model_info_list: List[ModelInfo]
        if parameters.save_weights and parameters.log_path is not None:
            model_info_list = ExperimentBase.save_model_paths(dataset.get_dataset_name(), models)
        else:
            model_info_list = ExperimentBase.get_empty_model_paths(models)

        result = ExperimentResult(self, TrainHistory(parameters, train_history_model_list, self.get_experiment_type(),
                                                     dataset.get_dataset_name()),
                                  model_info_list)

        if parameters.log_path is not None:
            ExperimentBase.save_experiment_result_to_file(parameters.log_path, result)
        return result


class ExperimentAutoencoder(ExperimentBase):

    def __init__(self, dataset_provider: DatasetProvider, model_provider: ModelProviderBase,
                 train_data_provider: TrainingDataProvider):
        super().__init__(dataset_provider, model_provider, train_data_provider)

    def train(self, parameters: BasicTrainParametersAutoencoder):
        early_stopping: CustomEarlyStopping = ExperimentAutoencoder.create_early_stopping(
            monitor=constants.Metrics.VAL_LOSS if parameters.validate else constants.Metrics.LOSS,
            patience=parameters.patience,
            min_delta=parameters.min_delta)
        callbacks = [early_stopping]
        dataset: Dataset = self.dataset_provider()

        models: Models = self.model_provider(dataset=dataset)
        train_data: BasicTrainingData = self.train_data_provider(dataset=dataset,
                                                                 train_data_rate=parameters.train_data_rate)

        autoencoder = models.autoencoder
        validation_data = train_data.validation_data if parameters.validate else None

        history_autoencoder: TrainHistoryModel = ExperimentBase.create_train_history_model_from_training(
            autoencoder.fit(x=train_data.x, y=train_data.y,
                            validation_data=validation_data, callbacks=callbacks,
                            epochs=parameters.epochs, batch_size=parameters.batch_size, verbose=2),
            early_stopping, parameters
        )
        result = self.process_experiment_training(dataset, parameters, models, [history_autoencoder])
        return models, result

    def get_experiment_type(self) -> str:
        return 'Experiment autoencoder solo'

    @staticmethod
    def create_early_stopping(monitor: str, patience: int, min_delta: float) -> CustomEarlyStopping:
        return CustomEarlyStopping(monitor=monitor, mode='min', verbose=1,
                                   patience=patience, min_delta=min_delta,
                                   restore_best_weights=True)

    @staticmethod
    def evaluate_on_test(dataset: Dataset, models: Models):
        evaluation = models.autoencoder.evaluate(x=dataset.get_test_images(), y=dataset.get_test_images(), verbose=2)
        print(evaluation)
        return evaluation

    @staticmethod
    def evaluate_on_train(dataset: Dataset, models: Models):
        evaluation = models.autoencoder.evaluate(x=dataset.get_train_images(), y=dataset.get_train_images(), verbose=2)
        print(evaluation)
        return evaluation

    @staticmethod
    def predict_test(dataset: Dataset, models: Models):
        predictions = models.autoencoder.predict(dataset.get_test_images(), verbose=2)
        return predictions

    @staticmethod
    def predict_train(dataset: Dataset, models: Models):
        predictions = models.autoencoder.predict(dataset.get_train_images(), verbose=2)
        return predictions

    @staticmethod
    def predicate_for_choosing_best_model():
        return ExperimentBase.create_predicate_for_choosing_best_model(constants.Models.AUTOENCODER,
                                                                       constants.Metrics.VAL_LOSS)


class ExperimentClassifier(ExperimentBase):

    def __init__(self, dataset_provider: DatasetProvider, model_provider: ModelProviderBase,
                 train_data_provider: TrainingDataProvider):
        super().__init__(dataset_provider, model_provider, train_data_provider)

    def train(self, parameters: BasicTrainParametersClassifier):
        early_stopping = ExperimentClassifier.create_early_stopping(monitor=constants.Metrics.VAL_ACCURACY,
                                                                    patience=parameters.patience,
                                                                    min_delta=parameters.min_delta)
        callbacks = [early_stopping]
        dataset: Dataset = self.dataset_provider()

        models: Models = self.model_provider(dataset=dataset)
        train_data: BasicTrainingData = self.train_data_provider(dataset=dataset,
                                                                 train_data_rate=parameters.train_data_rate)

        classifier = models.classifier
        validation_data = train_data.validation_data if parameters.validate else None

        models.set_encoder_layers_trainable(parameters.autoencoder_layers_trainable_during_classifier_training)
        print('Classifier trainable weights: ', len(classifier.trainable_weights))
        history_classifier: TrainHistoryModel = ExperimentBase.create_train_history_model_from_training(
            classifier.fit(x=train_data.x, y=train_data.y,
                           validation_data=validation_data, callbacks=callbacks,
                           epochs=parameters.epochs, batch_size=parameters.batch_size, verbose=2),
            early_stopping, parameters
        )
        models.set_encoder_layers_trainable(True)
        result = self.process_experiment_training(dataset, parameters, models, [history_classifier])
        return models, result

    def get_experiment_type(self) -> str:
        return 'Experiment classifier solo'

    @staticmethod
    def create_early_stopping(monitor: str, patience: int, min_delta: float) -> CustomEarlyStopping:
        return CustomEarlyStopping(monitor=monitor, mode='max', verbose=1,
                                   patience=patience, min_delta=min_delta,
                                   restore_best_weights=True)

    @staticmethod
    def predicate_for_choosing_best_model():
        return ExperimentBase.create_predicate_for_choosing_best_model(constants.Models.CLASSIFIER,
                                                                       constants.Metrics.VAL_ACCURACY)

    @staticmethod
    def evaluate_on_test(dataset: Dataset, models: Models):
        evaluation = models.classifier.evaluate(x=dataset.get_test_images(),
                                                y=dataset.get_test_labels_one_hot(),
                                                verbose=2)
        print(evaluation)
        return evaluation

    @staticmethod
    def predict_test(dataset: Dataset, models: Models):
        predictions = models.classifier.predict(dataset.get_test_images(), verbose=2)
        predictions_out = predictions.copy()

        predictions = np.argmax(predictions, axis=1)
        #predictions = predictions.astype('uint8')
        correct = [predictions[i] == value for i, value in enumerate(dataset.get_test_labels())]
        print("Correct: {}".format(correct.count(True)))
        return predictions_out

    @staticmethod
    def predict_train(dataset: Dataset, models: Models):
        predictions = models.classifier.predict(dataset.get_train_images(), verbose=2)
        predictions_out = predictions.copy()

        predictions = np.argmax(predictions, axis=1)
        # predictions = predictions.astype('uint8')
        correct = [predictions[i] == value for i, value in enumerate(dataset.get_train_labels())]
        print("Correct: {}".format(correct.count(True)))
        return predictions_out


class ExperimentAutoencoderAndClassifier(ExperimentBase):

    def __init__(self, dataset_provider: DatasetProvider, model_provider: ModelProviderBase,
                 train_data_provider: TrainingDataProvider):
        super().__init__(dataset_provider, model_provider, train_data_provider)

    def train(self, parameters: BasicTrainParametersTwoModels):
        parameters_autoencoder = parameters.train_parameters_autoencoder
        parameters_classifier = parameters.train_parameters_classifier

        autoencoder_metric = constants.Metrics.VAL_LOSS if parameters_autoencoder.validate else constants.Metrics.LOSS
        early_stopping_autoencoder: CustomEarlyStopping = ExperimentAutoencoder.create_early_stopping(
            monitor=autoencoder_metric,
            patience=parameters_autoencoder.epochs,
            min_delta=parameters_autoencoder.min_delta
        )
        callbacks_autoencoder = [early_stopping_autoencoder]
        early_stopping_classifier: CustomEarlyStopping = ExperimentClassifier.create_early_stopping(
            monitor=constants.Metrics.VAL_ACCURACY,
            patience=parameters_classifier.epochs,
            min_delta=parameters_classifier.min_delta
        )
        callbacks_classifier = [early_stopping_classifier]
        dataset: Dataset = self.dataset_provider()

        models: Models = self.model_provider(dataset=dataset)
        train_data: TrainingDataAutoencoderClassifier = self.train_data_provider(dataset=dataset,
                                                                                 train_data_rate_autoencoder=parameters_autoencoder.train_data_rate,
                                                                                 train_data_rate_classifier=parameters_classifier.train_data_rate)
        autoencoder = models.autoencoder
        classifier = models.classifier

        train_history_autoencoder: TrainHistoryModel = TrainHistoryModel.init_before_training(autoencoder.name,
                                                                                              early_stopping_autoencoder.monitor)
        train_history_classifier: TrainHistoryModel = TrainHistoryModel.init_before_training(classifier.name,
                                                                                             early_stopping_classifier.monitor)

        validation_data_autoencoder = train_data.training_data_autoencoder.validation_data if parameters_autoencoder.validate else None
        validation_data_classifier = train_data.training_data_classifier.validation_data if parameters_classifier.validate else None

        class WeightInfo:

            def __init__(self, monitor_value, weights_autoencoder, weights_classifier , epoch) -> None:
                self.monitor_value = monitor_value
                self.weights_autoencoder =weights_autoencoder
                self. weights_classifier =  weights_classifier
                self.epoch = epoch

        weight_info_autoencoder: WeightInfo = WeightInfo(0, None, None, -1)
        weight_info_classifier: WeightInfo = WeightInfo(0, autoencoder.get_weights(), classifier.get_weights(), -1)

        for cycle in range(parameters.training_cycles):
            print('Cycle:', cycle + 1)
            if parameters_autoencoder.epochs is not None and parameters_autoencoder.epochs > 0:
                epoch_autoencoder_start = cycle * parameters_autoencoder.epochs + 1

                local_history_autoencoder = autoencoder.fit(x=train_data.training_data_autoencoder.x,
                                                            y=train_data.training_data_autoencoder.y,
                                                            validation_data=validation_data_autoencoder,
                                                            callbacks=callbacks_autoencoder,
                                                            epochs=parameters_autoencoder.epochs,
                                                            batch_size=parameters_autoencoder.batch_size,
                                                            verbose=2)
                loss_values_autoencoder = local_history_autoencoder.history[early_stopping_autoencoder.monitor]
                best_epoch_index = loss_values_autoencoder.index(min(loss_values_autoencoder))
                if weight_info_autoencoder.monitor_value > early_stopping_autoencoder.best:
                    weight_info_autoencoder.monitor_value = early_stopping_autoencoder.best
                    weight_info_autoencoder.epoch = epoch_autoencoder_start + best_epoch_index
                    weight_info_autoencoder.weights = autoencoder.get_weights()

                train_history_autoencoder.add_history_dict(local_history_autoencoder.history)

            if True:  # aby sa mi nahodou nebili premenne
                models.set_encoder_layers_trainable(
                    parameters.autoencoder_layers_trainable_during_classifier_training)
                epoch_classifier_start = cycle * parameters_classifier.epochs + 1
                local_history_classifier = classifier.fit(x=train_data.training_data_classifier.x,
                                                          y=train_data.training_data_classifier.y,
                                                          validation_data=validation_data_classifier,
                                                          callbacks=callbacks_classifier,
                                                          epochs=parameters_classifier.epochs,
                                                          batch_size=parameters_classifier.batch_size,
                                                          verbose=2)
                loss_values_classifier = local_history_classifier.history[early_stopping_classifier.monitor]
                best_epoch_index = loss_values_classifier.index(min(loss_values_classifier))
                if weight_info_classifier.monitor_value < early_stopping_classifier.best:
                    weight_info_classifier.monitor_value = early_stopping_classifier.best
                    weight_info_classifier.epoch = epoch_classifier_start + best_epoch_index
                    weight_info_classifier.weights_autoencoder = autoencoder.get_weights()
                    weight_info_classifier.weights_classifier = classifier.get_weights()

                train_history_classifier.add_history_dict(local_history_classifier.history)
                models.set_encoder_layers_trainable(True)

        autoencoder.set_weights(weight_info_classifier.weights_autoencoder)
        classifier.set_weights(weight_info_classifier.weights_classifier)

        train_history_autoencoder.set_best_epoch(weight_info_autoencoder.epoch)
        train_history_autoencoder.set_monitor_best_value(weight_info_autoencoder.monitor_value)
        train_history_autoencoder.set_stopped_epoch(parameters_autoencoder.epochs * parameters.training_cycles)

        train_history_classifier.set_best_epoch(weight_info_classifier.epoch)
        train_history_classifier.set_monitor_best_value(weight_info_classifier.monitor_value)
        train_history_classifier.set_stopped_epoch(parameters_classifier.epochs * parameters.training_cycles)

        result = self.process_experiment_training(dataset, parameters, models,
                                                  [train_history_autoencoder, train_history_classifier])
        return models, result

    def get_experiment_type(self) -> str:
        return 'Experiment classifier autoencoder'


class ExperimentAutoClassifier(ExperimentBase):

    def __init__(self, dataset_provider: DatasetProvider, model_provider: ModelProviderBase,
                 train_data_provider: TrainingDataProvider):
        super().__init__(dataset_provider, model_provider, train_data_provider)

    def train(self, parameters: BasicTrainParametersAutoClassifier):
        early_stopping: CustomEarlyStopping = ExperimentClassifier.create_early_stopping(
            monitor=constants.Metrics.VAL_CLASSIFIER_OUT_ACCURACY,
            patience=parameters.epochs,
            min_delta=parameters.min_delta
        )
        callbacks = [early_stopping]
        dataset: Dataset = self.dataset_provider()

        models: Models = self.model_provider(dataset=dataset)
        if parameters.loss_weigh_decoder is not None and parameters.loss_weight_classifier is not None:
            models.compile_models(parameters.loss_weigh_decoder, parameters.loss_weight_classifier)

        train_data: BasicTrainingData = self.train_data_provider(dataset=dataset,
                                                                 train_data_rate=parameters.train_data_rate)

        validation_data = train_data.validation_data if parameters.validate else None

        auto_classifier = models.auto_classifier

        history_autoclassifier: TrainHistoryModel = ExperimentBase.create_train_history_model_from_training(
            auto_classifier.fit(x=train_data.x, y=train_data.y, validation_data=validation_data,
                                batch_size=parameters.batch_size, epochs=parameters.epochs,
                                callbacks=callbacks, verbose=2),
            early_stopping, parameters)

        result = self.process_experiment_training(dataset, parameters, models,
                                                  [history_autoclassifier])
        return models, result

    @staticmethod
    def predicate_for_choosing_best_model():
        return ExperimentBase.create_predicate_for_choosing_best_model(constants.Models.AUTO_CLASSIFIER,
                                                                       constants.Metrics.VAL_CLASSIFIER_OUT_ACCURACY)

    def get_experiment_type(self) -> str:
        return 'Experiment autoclassifier'

    @staticmethod
    def evaluate_on_test(dataset: Dataset, models: Models):
        evaluation = models.auto_classifier.evaluate(x=dataset.get_test_images(),
                                                     y={constants.Models.DECODED_OUT: dataset.get_test_images(),
                                                        constants.Models.CLASSIFIER_OUT: dataset.get_test_labels_one_hot()},
                                                     verbose=2)
        print(evaluation)
        return evaluation

    @staticmethod
    def predict_test(dataset: Dataset, models: Models):
        predictions_auto_encoder, predictions_classifier = models.auto_classifier.predict(dataset.get_test_images(),
                                                                                          verbose=2)
        predictions_classifier_out: np.ndarray = predictions_classifier.copy()
        predictions_classifier = np.argmax(np.round(predictions_classifier), axis=1)
        correct = np.where(predictions_classifier == dataset.get_test_labels())[0]
        print("Found {} correct labels".format(len(correct)))
        return predictions_auto_encoder, predictions_classifier_out


def remove_models_without_log():
    import global_functions
    all_h5 = global_functions.get_files_in_dir_with_extension(constants.Paths.OUTPUT_DIRECTORY, '.h5')
    all_json = global_functions.get_files_in_dir_with_extension(constants.Paths.OUTPUT_DIRECTORY, '.json')
    for log_file in all_json:
        results = ExperimentBase.load_experiment_results(log_file)
        for result in results:
            for model_info in result.model_infos:
                all_h5.remove(model_info.path_to_model)
    print(len(all_h5))
    for to_delete in all_h5:
        print(to_delete)
        #global_functions.remove_file(to_delete)