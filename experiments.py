from abc import ABC, abstractmethod

from tensorflow import keras
from keras.models import Model

# from global_functions import run_once
from constants import Constants
from typing import List, Dict, Callable
from datasets import Dataset
import collections

from models import Models
from functools import wraps
import json
import codecs
from pathlib import Path
from datetime import datetime
from pathlib import Path

class Logger():
    AUTOENCODE_OUTPUTH_PATH = 'model_autoencoder_log.json'

    ROOT_KEY = 'modellist'
    MODEL_KEY = 'model'
    HISTORY_NODE_KEY = 'training_history'
    HISTORY_KEY = 'history'
    PARAMS_KEY = 'params'
    MODEL_PATH = 'path_to_model'

    def __init__(self, logfile=Constants.Paths.OUTPUT_DIRECTORY + 'model_default_log.json', save_whole_model=False):
        self.logfile = logfile
        self.save_whole_model = save_whole_model

    @staticmethod
    def convert_dictionary(dictionary):
        new_hist = {}
        for key in list(dictionary.keys()):
            if type(dictionary[key]) == np.ndarray:
                new_hist[key] = dictionary[key].tolist()
            elif type(dictionary[key]) == list:
                if type(dictionary[key][0]) in (np.float64, np.float32):
                    new_hist[key] = list(map(float, dictionary[key]))
                else:
                    new_hist[key] = dictionary[key]
            else:
                if type(dictionary[key]) in (np.int32, np.int64):
                    new_hist[key] = int(dictionary[key])
                else:
                    new_hist[key] = dictionary[key]
        return new_hist

    @staticmethod
    def serialize_history(history):
        history_dict = Logger.convert_dictionary(history.history)
        params_dict = Logger.convert_dictionary(history.params)
        return {Logger.HISTORY_KEY: history_dict, Logger.PARAMS_KEY: params_dict}

    def get_model_path(self):
        now = datetime.now()
        return OUTPUT_DIRECTORY + '_' + now.strftime("_%d_%m_%Y_%H_%M_%S") + '.h5'

    @staticmethod
    def get_log(logfile, initialize_if_empty=True):
        print('Logfile: ' + logfile)
        default_dict = {Logger.ROOT_KEY: []}
        path_to_logfile = Path(logfile)
        if Path(path_to_logfile).exists():
            text = Path(path_to_logfile).read_text()
            return json.loads(text)
        else:
            if initialize_if_empty:
                return default_dict
            else:
                None

    def log_history(self, history, logfile=None):
        if logfile is None:
            logfile = self.logfile
        model_to_dictionary = history.model.get_config()
        #             # Open the logfile and append
        result_dictionary = Logger.get_log(self.logfile)
        with codecs.open(self.logfile, 'w', encoding='UTF-8') as opened_file:
            # Now we log to the specified logfile
            history_dict = Logger.serialize_history(history)
            model_path = None
            if self.save_whole_model:
                model_path = self.get_model_path()
                history.model.save(model_path)
            current_history = {Logger.MODEL_KEY: model_to_dictionary, Logger.HISTORY_NODE_KEY: history_dict,
                               Logger.MODEL_PATH: model_path}
            # current_history = {MODEL_KEY : 'model', PARAMS_KEY: 'parameter', TRAIN_HISTORY_KEY: 'trening' }
            result_dictionary[Logger.ROOT_KEY].append(current_history)
            opened_file.write(json.dumps(result_dictionary))
            # opened_file.write(str(result_dictionary))
        print("zapisane")

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            print('fungujem')
            history = func(*args, **kwargs)
            self.log_history(history, self.logfile)
            return history

        return wrapped_function

    @staticmethod
    def get_models_from_log(log):
        modellist = log[Logger.ROOT_KEY]
        result_list = []
        for model_info_dict in modellist:
            model_path = model_info_dict[Logger.MODEL_PATH];
            if (model_path is None):
                result_list.append(None)
            else:
                result_list.append(keras.models.load_model(model_path))
        return result_list


class Vizualizer():

    @staticmethod
    def vizualize_history(history_dict, ignore=0.2, metrics=None):
        metrics = [metric.value for metric in metrics]
        lines = []
        epochs = history_dict[Logger.PARAMS_KEY]['epochs']
        start_index = int(round(ignore * epochs))
        x = range(start_index, epochs)
        plt.figure(figsize=(10, 10))
        for metric, values in history_dict[Logger.HISTORY_KEY].items():
            # if next((s for s in metrics if metric in s.lower()), None) is not None:
            y = values[start_index:]
            plt.plot(x, y, 'ro', label=metric)
            line, = plt.plot(x, y, label=metric)

            for i_x, i_y in zip(x, y):
                print(i_x, i_y)
                plt.text(i_x, i_y, '{0:.4f}'.format(round(i_y, 4)))

            lines.append(line)
            print('metric: ' + metric)
            print("Values: ", values)
        plt.legend(handles=lines)
        plt.xlabel('Epoch')
        plt.xticks(range(epochs))

        plt.show()

    @staticmethod
    def vizualize(log_dictionary, ignore=0.2, metrics=None):
        modellist = log_dictionary[Logger.ROOT_KEY]
        for model_index, model_info_dict in enumerate(modellist):
            print('Model index: ' + str(model_index + 1))
            model = model_info_dict[Logger.MODEL_KEY]
            history_dict = model_info_dict[Logger.HISTORY_NODE_KEY]
            Vizualizer.vizualize_history(history_dict, ignore, metrics)


class ModelProvider(ABC):

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    @abstractmethod
    def provide_model(self) -> Models:
        pass


class BasicModelProvider(ModelProvider):

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def provide_model(self) -> Model:
        models = BasicModelBuilder.from_dataset(dataset).create_models()
        return models.autoencoder


def basic_model_provider(dataset: Dataset):
    return BasicModelBuilder.from_dataset(dataset).create_models()


from global_functions import auto_str_repr


@auto_str_repr
class TrainParameters(ABC):

    def to_dict(self) -> Dict:
        return self.__dict__





@auto_str_repr
class TrainHistory:

    def __init__(self, params: TrainParameters):
        self.metrics = set()
        self.values = {}
        self.params = params

    def add_metric_value(self, metric, value):
        self.metrics.add(metric)
        list_of_values = self.values.get(metric, [])
        list_of_values.extend(list(value))
        self.values[metric] = list_of_values

    def add_history_dict(self, history: Dict):
        for metric, value in history.items():
            self.add_metric_value(metric, value)


@auto_str_repr
class BasicTrainParameters(TrainParameters):

    def __init__(self, epochs: int, batch_size: int, validate: bool, log_path: str, save_weights: bool) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.validate = validate
        self.log_path = log_path
        self.save_weights = save_weights

@auto_str_repr
class ModelInfo():

    def __init__(self, path_to_model: str) -> None:
        self.path_to_model = path_to_model


ModelProvider = Callable[..., Models]
DatasetProvider = Callable[[], Dataset]


@auto_str_repr
class ExperimentResult:

    def __init__(self, dataset_provider: DatasetProvider, model_provider: ModelProvider,
                 train_history: List[TrainHistory], model_infos: List[ModelInfo]):
        self.dataset_provider = dataset_provider
        self.model_provider = model_provider
        self.train_history = train_history
        self.model_infos = model_infos


import jsonpickle


class ExperimentAutoencoder:

    def __init__(self, model_provider: ModelProvider, dataset_provider: DatasetProvider):
        self.model_provider: ModelProvider = model_provider
        self.dataset_provider: DatasetProvider = dataset_provider
        self.train_images = None
        self.validation_images = None

    def prepare_data_for_training(self, dataset: Dataset):
        self.train_images = dataset.get_train_images()[:10000]
        self.validation_images = dataset.get_test_images()

    @staticmethod
    def get_model_path(model_name : str):
        now = datetime.now()
        return Constants.Paths.OUTPUT_DIRECTORY + '_' + model_name + now.strftime("_%d_%m_%Y_%H_%M_%S") + '.h5'

    @staticmethod
    def load_experiment_result(log_file):
        text = Path(log_file).read_text()
        experiment: ExperimentResult = jsonpickle.decode(text)
        def provide_existing_model(dataset: Dataset) -> Models:
            model_list = [keras.models.load_model(model_info.path_to_model) for model_info in experiment.model_infos]
            models: Models = Models(*model_list)
            return models
        return ExperimentAutoencoder(provide_existing_model, experiment.dataset_provider)


    def train(self, parameters: BasicTrainParameters):
        early_stopping = keras.callbacks.EarlyStopping(monitor=Constants.Metrics.VAL_LOSS, mode='min', verbose=1,
                                                       patience=10, min_delta=0.005,
                                                       restore_best_weights=True)
        callbacks = [early_stopping]
        dataset: Dataset = self.dataset_provider()

        self.prepare_data_for_training(dataset)

        models: Models = self.model_provider(dataset)
        autoencoder = models.autoencoder

        my_history: TrainHistory = TrainHistory(parameters)
        validation_data = None
        if parameters.validate:
            validation_data = (self.validation_images, self.validation_images)
        history = autoencoder.fit(x=self.train_images, y=self.train_images,
                                  validation_data=validation_data, callbacks=callbacks,
                                  epochs=parameters.epochs, batch_size=parameters.batch_size)

        my_history.add_history_dict(history.history)

        print(my_history)
        print("loss", my_history.values['loss'])

        model_paths = []
        for model in models:
            model_path = ExperimentAutoencoder.get_model_path(model.name)
            model_paths.append(model_path)
            model.save(model_path)

        result = ExperimentResult(self.dataset_provider, self.model_provider, [my_history],
                                  [ModelInfo(model_path) for model_path in model_paths])

        if parameters.log_path is not None:
            encoded = jsonpickle.encode(result, keys=True)
            with open(parameters.log_path, 'w') as file:
                file.write(encoded)
        return models, result

    def get_experiment_type(self):
        return 'Type'


from models import BasicModelBuilder
from datasets import FashionMnistDataset
from global_functions import on_start

on_start()

dataset = FashionMnistDataset()


def provide_dataset() -> Dataset:
    return FashionMnistDataset()


class DatasetProviderClass():

    def __init__(self, provider: DatasetProvider):
        self.provider: DatasetProvider = provider

    def __call__(self) -> Dataset:
        return self.provider()


#exp = ExperimentAutoencoder(basic_model_provider, DatasetProviderClass(FashionMnistDataset))
#exp.train(BasicTrainParameters(1, 128, True, 'mylog.json', True))


#print(experiment.model_provider(dataset))
#print('dataset provider', experiment.dataset_provider)
#print(experiment)

exp_loaded : ExperimentAutoencoder = ExperimentAutoencoder.load_experiment_result('mylog2.json')
exp_loaded.model_provider(None)
exp_loaded.train(BasicTrainParameters(10, 128, True, 'mylog3.json', True))
exp_loaded.train(BasicTrainParameters(1, 128, True, 'mylog3.json', True))


