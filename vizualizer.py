from experiments import TrainHistory
from experiments import TrainHistoryModel
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset

class Vizualizer():

    @staticmethod
    def vizualize_history(train_history_model: TrainHistoryModel, ignore: float):
        print('Model name: ', train_history_model.model_name)
        print('Stopped epoch: ', train_history_model.stopped_epoch)
        print('Best epoch: ', train_history_model.best_epoch)
        print('Value to monitor: ', train_history_model.value_to_monitor)
        print('Monitor best value: ', train_history_model.monitor_best_value)
        epochs = train_history_model.stopped_epoch
        if epochs == 1:
            print('stoj')
        lines = []
        plt.figure(figsize=(10, 10))
        start_index = int(round(ignore * epochs))
        for metric, values in train_history_model.values.items():
            # if next((s for s in metrics if metric in s.lower()), None) is not None:

            x = range(start_index, epochs)
            y = values[start_index:]
            plt.plot(x, y, 'ro', label=metric)
            line, = plt.plot(x, y, label=metric)

            for i_x, i_y in zip(x, y):
                plt.text(i_x, i_y, '{0:.4f}'.format(round(i_y, 4)))
            lines.append(line)
            print('metric: ' + metric)
            print("Values: ", values)

        plt.legend(handles=lines)
        plt.xticks(range(max(start_index - 1, 0), epochs))
        plt.xlabel('Epoch')
        plt.title(train_history_model.model_name)
        plt.show()

    @staticmethod
    def vizualize(train_history: List[TrainHistory], ignore: float):
        for model_index, train_history in enumerate(train_history):
            print('Experiment type', train_history.experiment_type)
            print('Model index: ', model_index + 1)
            print("Parameters: ", train_history.train_parameters)
            for train_history_concrete in train_history.train_history_model_list:
                Vizualizer.vizualize_history(train_history_concrete, ignore)

    @staticmethod
    def show_autoencoder_images(dataset: Dataset, test_images, predictions, number_of_images):
        #np.random.seed(42)
        random_test_images = np.random.randint(test_images.shape[0], size=number_of_images)
        plt.figure(figsize=(28, 16))
        for i, image_idx in enumerate(random_test_images):
            # plot original image
            ax = plt.subplot(2, number_of_images, i + 1)
            plt.title("Original")
            plt.imshow(dataset.process_image_for_plotting(test_images[image_idx]) )
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, number_of_images, number_of_images + i + 1)
            plt.imshow(dataset.process_image_for_plotting(predictions[image_idx]) )  # reshape(28, 28)
            plt.title("Predicted")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    @staticmethod
    def plot_bins(predictions, number_of_bins):
        import numpy as np
        import matplotlib.pyplot as plt
        bins = np.linspace(predictions.min() - 0.01, predictions.max() + 0.01, num=number_of_bins)

        print(bins)
        fig, ax = plt.subplots(1, 1)
        hist = ax.hist(predictions, bins=bins)
        labels = [str(count) for count in hist[0]]
        # labels.insert(0, 0)
        print(labels)
        # ax.set_xticklabels(labels)
        plt.show()
        return bins, labels