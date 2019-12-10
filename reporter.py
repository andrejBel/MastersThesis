from typing import List
from experiments import ExperimentResult, ExperimentClassifier, ExperimentAutoencoder, ExperimentAutoClassifier, \
    ExperimentAutoencoderAndClassifier
from experiments import BasicTrainParametersClassifier, BasicTrainParametersAutoencoder, BasicTrainParametersTwoModels, \
    BasicTrainParametersAutoClassifier
from openpyxl import Workbook
from openpyxl import load_workbook
from pathlib import Path
import constants


class Reporter():

    @staticmethod
    def create_classifier_report(experiment_results: List[ExperimentResult], path_to_workbook, experiment_type):
        wb: Workbook = None
        if Path(path_to_workbook).exists():
            wb = load_workbook(path_to_workbook)
        else:
            wb = Workbook()
        ws = wb.active
        ws.append(["Typ experimentu", experiment_type])
        ws.append(
            ['Vrstvy enkódera trénovateľné',
             "Percento trénovacích dát",
             "Počet epoch",
             "Najlepšia epocha",
             "Presnosť na testovacom datasete"])
        for result in experiment_results:
            train_history: BasicTrainParametersClassifier = result.train_history.train_parameters
            rate = train_history.train_data_rate
            encoder_trainable = train_history.autoencoder_layers_trainable_during_classifier_training
            encoder_trainable_str = "Áno" if encoder_trainable else "Nie"
            for model in result.train_history.train_history_model_list:
                if model.model_name == constants.Models.CLASSIFIER:
                    ws.append([encoder_trainable_str,
                               str(round(rate * 100, 0)).replace('.', ','),
                               model.stopped_epoch,
                               model.best_epoch,
                               str(model.monitor_best_value).replace('.', ',')])
        wb.save(path_to_workbook)

    @staticmethod
    def create_autoencoder_report(experiment_results: List[ExperimentResult], path_to_workbook, experiment_type):
        wb: Workbook = None
        if Path(path_to_workbook).exists():
            wb = load_workbook(path_to_workbook)
        else:
            wb = Workbook()
        ws = wb.active
        ws.append(["Typ experimentu", experiment_type])
        ws.append(
            ["Percento trénovacích dát",
             "Počet epoch",
             "Najlepšia epocha",
             "Rekonštrukčná chyba"])
        for result in experiment_results:
            train_history: BasicTrainParametersAutoencoder = result.train_history.train_parameters
            rate = train_history.train_data_rate
            for model in result.train_history.train_history_model_list:
                if model.model_name == constants.Models.AUTOENCODER:
                    ws.append([
                        str(round(rate * 100, 0)).replace('.', ','),
                        model.stopped_epoch,
                        model.best_epoch,
                        str(model.monitor_best_value).replace('.', ',')])
        wb.save(path_to_workbook)

    @staticmethod
    def create_autoencoder_classifier_together_report(experiment_results: List[ExperimentResult], path_to_workbook,
                                                      experiment_type):
        wb: Workbook = None
        if Path(path_to_workbook).exists():
            wb = load_workbook(path_to_workbook)
        else:
            wb = Workbook()
        ws = wb.active
        ws.append(["Typ experimentu", experiment_type])
        ws.append(
            ['Vrstvy enkódera trénovateľné',
             "Percento trénovacích dát autoenkóder",
             "Percento trénovacích dát klasifikátor",
             "Rekonštrukčná chyba",
             "Presnosť na testovacom datasete"])
        for result in experiment_results:
            train_history: BasicTrainParametersTwoModels = result.train_history.train_parameters
            rate_autoencoder = train_history.train_parameters_autoencoder.train_data_rate
            rate_classifier = train_history.train_parameters_classifier.train_data_rate
            encoder_trainable = train_history.autoencoder_layers_trainable_during_classifier_training
            encoder_trainable_str = "Áno" if encoder_trainable else "Nie"
            classifier_best_value = None
            autoencoder_best_value = None
            for model in result.train_history.train_history_model_list:

                if model.model_name == constants.Models.CLASSIFIER:
                    classifier_best_value = model.monitor_best_value
                if model.model_name == constants.Models.AUTOENCODER:
                    autoencoder_best_value = model.monitor_best_value
            ws.append([encoder_trainable_str,
                       str(round(rate_autoencoder * 100, 0)).replace('.', ','),
                       str(round(rate_classifier * 100, 0)).replace('.', ','),
                       str(autoencoder_best_value).replace('.', ','),
                       str(classifier_best_value).replace('.', ',')])
        wb.save(path_to_workbook)

    @staticmethod
    def create_autoclassifier_report(experiment_results: List[ExperimentResult], path_to_workbook,
                                     experiment_type):
        wb: Workbook = None
        if Path(path_to_workbook).exists():
            wb = load_workbook(path_to_workbook)
        else:
            wb = Workbook()
        ws = wb.active
        ws.append(["Typ experimentu", experiment_type])
        ws.append(
            ["Percento trénovacích dát",
             "Presnosť na testovacom datasete"])
        for result in experiment_results:
            train_history: BasicTrainParametersAutoClassifier = result.train_history.train_parameters
            rate = train_history.train_data_rate
            for model in result.train_history.train_history_model_list:
                if model.model_name == constants.Models.AUTO_CLASSIFIER:
                    ws.append([
                        str(round(rate * 100, 0)).replace('.', ','),
                        str(model.monitor_best_value).replace('.', ',')])
        wb.save(path_to_workbook)

    @staticmethod
    def create_report(experiment_results: List[ExperimentResult], path_to_workbook,
                      experiment_type):
        if len(experiment_results) > 0:
            result: ExperimentResult = experiment_results[0]
            experiment = result.experiment
            if isinstance(experiment, ExperimentClassifier):
                Reporter.create_classifier_report(experiment_results, path_to_workbook, experiment_type)
            elif isinstance(experiment, ExperimentAutoencoder):
                Reporter.create_autoencoder_report(experiment_results, path_to_workbook, experiment_type)
            elif isinstance(experiment, ExperimentAutoencoderAndClassifier):
                Reporter.create_autoencoder_classifier_together_report(experiment_results, path_to_workbook,
                                                                       experiment_type)
            elif isinstance(experiment, ExperimentAutoClassifier):
                Reporter.create_autoclassifier_report(experiment_results, path_to_workbook,
                                                      experiment_type)
            else:
                raise Exception("Unknown experiment type")
