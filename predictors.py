from abc import ABC, abstractmethod

import numpy as np

import constants
import datasets
import experiments
from datasets import Dataset
from excel_logger import ExcelLogger
from models import Models


class Predictor(ABC):

    def predict_train(self, dataset: Dataset, model: Models, threashold=None):
        return self.predict(dataset.get_train_images(), dataset.get_train_labels(), model, threashold)

    def predict_test(self, dataset, model, threashold=None):
        return self.predict(dataset.get_test_images(), dataset.get_test_labels(), model, threashold)

    def predict_test_based_on_train_threashoald(self, dataset, model, percentile):
        print("Predict train")
        ExcelLogger.set_logging(False)
        _, distance_train = self.predict_train(dataset, model)
        threashold_for_test = np.percentile(distance_train, percentile)
        print("Predict test")
        ExcelLogger.set_logging(True)
        result = self.predict_test(dataset, model, threashold_for_test), threashold_for_test
        ExcelLogger.set_logging(False)
        return result

    @abstractmethod
    def predict(self, images: np.ndarray, labels: np.ndarray, model: Models, threashold):
        pass

    @staticmethod
    def predict_for_subclasses(dataset_for_threashold: Dataset, dataset_for_test: Dataset, model: Models, percentile,
                               path_to_workbook):
        import global_functions
        predictors = global_functions.get_all_subclasses(Predictor)
        ExcelLogger.init_logger(path_to_workbook)
        ExcelLogger.write_into_sheet(["Train dataset", dataset_for_threashold.get_dataset_name()], True)
        ExcelLogger.write_into_sheet(["Test dataset", dataset_for_test.get_dataset_name()], True)
        ExcelLogger.write_into_sheet(
            ["Predictor", "Correct", "Incorrect", "Unknown", "Correct", "Incorrect", "Unknown"],
            True)
        for predictor in predictors:
            print(predictor.__name__)
            instance = predictor()
            ExcelLogger.set_logging(False)
            ExcelLogger.append_buffer(predictor.__name__, True)
            (_, __), threashoald = instance.predict_test_based_on_train_threashoald(dataset_for_threashold,
                                                                                    model, percentile)
            print(predictor.__name__)
            print("Threashold: ", threashoald)
            print("Test:")

            ExcelLogger.set_logging(True)
            instance.predict_test(dataset_for_test, model, threashoald)
            ExcelLogger.set_logging(False)
            ExcelLogger.flush()

    @staticmethod
    def predict_for_datasets(classifier_path: str, path_to_excel, train_dataset: datasets.Dataset,
                             test_dataset: datasets.Dataset, general_info_list, activation_func=None):
        import reporter
        results = experiments.ExperimentBase.load_experiment_results(
            classifier_path, True)
        experiments.ExperimentBase.sort_results_ascending_by_train_rate(results)
        experiment = results[-1].experiment
        trained_models_classifier = experiment.model_provider()
        # if activation_func is None:
        trained_models_classifier.make_classifier_without_activation()
        # elif activation_func == 'sigmoid':
        # trained_models_classifier.make_classifier_with_sigmoid_activation()
        # elif activation_func == 'softmax':
        ExcelLogger.append_to_workbook(general_info_list, path_to_excel)
        Predictor.predict_for_subclasses(train_dataset, test_dataset,
                                         trained_models_classifier,
                                         10, path_to_excel)


class AbsDistanceFromPredicted(Predictor):

    def predict(self, images, labels, model, threashold):
        predictions = model.classifier.predict(images)
        my_classes = predictions.argmax(axis=1)
        distances = np.zeros_like(my_classes)
        for index, _ in enumerate(predictions):
            classes_values = predictions[index]
            my_predicted_class = my_classes[index]
            distance = 0.0
            for index_class_value, class_value in enumerate(classes_values):
                if index_class_value == my_predicted_class:
                    continue
                distance += abs(classes_values[my_predicted_class] - class_value)
            if threashold is not None and distance < threashold:
                my_classes[index] = -1
            distances[index] = distance
        correct = (labels == my_classes).sum()
        unknown = (-1 == my_classes).sum()
        incorrect = len(images) - correct - unknown
        ExcelLogger.extend_buffer([correct, incorrect, unknown])
        print("correct: ", correct)
        print("incorrect: ", incorrect)
        print("unknown: ", unknown)
        return my_classes, distances


class SquereDistanceFromPredicted(Predictor):

    def predict(self, images, labels, model, threashold):
        predictions = model.classifier.predict(images)
        my_classes = predictions.argmax(axis=1)
        distances = np.zeros_like(my_classes)
        for index, _ in enumerate(predictions):
            classes_values = predictions[index]
            my_predicted_class = my_classes[index]
            distance = 0.0
            for index_class_value, class_value in enumerate(classes_values):
                if index_class_value == my_predicted_class:
                    continue

                distance += np.square(classes_values[my_predicted_class] - class_value)
            # distance /= (len(classes_values) - 1)
            if threashold is not None and distance < threashold:
                my_classes[index] = -1
            distances[index] = distance
        correct = (labels == my_classes).sum()
        unknown = (-1 == my_classes).sum()
        incorrect = len(images) - correct - unknown
        ExcelLogger.extend_buffer([correct, incorrect, unknown])
        print("correct: ", correct)
        print("incorrect: ", incorrect)
        print("unknown: ", unknown)
        return my_classes, distances


class SumAbs(Predictor):

    def predict(self, images, labels, model, threashold):
        predictions = model.classifier.predict(images)
        my_classes = predictions.argmax(axis=1)
        sums = np.zeros_like(my_classes)
        for index, _ in enumerate(predictions):
            classes_values = predictions[index]
            sum_values = np.sum(np.absolute(classes_values))
            if threashold is not None and sum_values < threashold:
                my_classes[index] = -1
            sums[index] = sum_values
        correct = (labels == my_classes).sum()
        unknown = (-1 == my_classes).sum()
        incorrect = len(images) - correct - unknown
        ExcelLogger.extend_buffer([correct, incorrect, unknown])
        print("correct: ", correct)
        print("incorrect: ", incorrect)
        print("unknown: ", unknown)
        return my_classes, sums


class SumSquere(Predictor):

    def predict(self, images, labels, model, threashold):
        predictions = model.classifier.predict(images)
        my_classes = predictions.argmax(axis=1)
        sums = np.zeros_like(my_classes)
        for index, _ in enumerate(predictions):
            classes_values = predictions[index]
            sum_values = np.sum(np.square(classes_values))
            if threashold is not None and sum_values < threashold:
                my_classes[index] = -1
            sums[index] = sum_values
        correct = (labels == my_classes).sum()
        unknown = (-1 == my_classes).sum()
        incorrect = len(images) - correct - unknown
        ExcelLogger.extend_buffer([correct, incorrect, unknown])
        print("correct: ", correct)
        print("incorrect: ", incorrect)
        print("unknown: ", unknown)
        return my_classes, sums


class MaxArgPredictor(Predictor):

    def predict(self, images, labels, model, threashold):
        predictions = model.classifier.predict(images)
        my_classes = predictions.argmax(axis=1)
        sums = np.zeros_like(my_classes)
        for index, _ in enumerate(predictions):
            class_value = predictions[index][my_classes[index]]
            if threashold is not None and class_value < threashold:
                my_classes[index] = -1
        correct = (labels == my_classes).sum()
        unknown = (-1 == my_classes).sum()
        incorrect = len(images) - correct - unknown
        ExcelLogger.extend_buffer([correct, incorrect, unknown])
        print("correct: ", correct)
        print("incorrect: ", incorrect)
        print("unknown: ", unknown)
        predictions = np.max(predictions, axis=1)
        return my_classes, predictions


# Predictor.predict_for_subclasses(None, None, None, None)


if __name__ == "__main__":
    import reporter

    Predictor.predict_for_datasets(
        constants.ExperimentsPaths.Mnist.BasicModelBuilder.CLASSIFIER_EN_LAYERS_ON,
        "example.xlsx",
        datasets.MnistDataset(),
        datasets.Cifar10GrayDataset(),
        ['Experiment 2 Basic model builder'])
