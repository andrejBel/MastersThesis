
from datasets import Dataset
from models import Models
from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
    def predict_train(self, dataset : Dataset, model : Models, threashold=None):
        return self.predict(dataset.get_train_images(), dataset.get_train_labels(), model, threashold)

    def predict_test(self, dataset, model, threashold=None):
        return self.predict(dataset.get_test_images(), dataset.get_test_labels(), model, threashold)

    def predict_test_based_on_train_treshoald(self, dataset, model, percentile):
        print("Predict train")
        _, distance_train = self.predict_train(dataset, model)
        threashold_for_test = np.percentile(distance_train, percentile)
        print("Predict test")
        return self.predict_test(dataset, model, threashold_for_test), threashold_for_test

    @abstractmethod
    def predict(self, images : np.ndarray, labels : np.ndarray, model : Models, threashold):
        pass

    @staticmethod
    def predict_for_subclasses(dataset_for_threashold : Dataset, dataset_for_test, model: Models, percentile):
        import global_functions
        predictors = global_functions.get_all_subclasses(Predictor)
        threashoald_list = []
        for predictor in predictors:
            print(predictor.__name__)
            instance = predictor()

            (_, __), threashoald = instance.predict_test_based_on_train_treshoald(dataset_for_threashold,
                                                                                  model, percentile)
            threashoald_list.append(threashoald)
        for index, predictor in enumerate(predictors):
            print(predictor.__name__)
            instance = predictor()
            print("Threashold: ", threashoald_list[index])
            print("Test:")
            instance.predict_test(dataset_for_test, model, threashoald_list[index])

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
        print("correct: ", (labels == my_classes).sum())
        print("unknown: ", (-1 == my_classes).sum())
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
        print("correct: ", (labels == my_classes).sum())
        print("unknown: ", (-1 == my_classes).sum())
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
        print("correct: ", (labels == my_classes).sum())
        print("unknown: ", (-1 == my_classes).sum())
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
        print("correct: ", (labels == my_classes).sum())
        print("unknown: ", (-1 == my_classes).sum())
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
        print("correct: ", (labels == my_classes).sum())
        print("unknown: ", (-1 == my_classes).sum())
        predictions = np.max(predictions, axis=1)
        return my_classes, predictions


#Predictor.predict_for_subclasses(None, None, None, None)