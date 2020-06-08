from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset
from experiments import TrainHistory
from experiments import TrainHistoryModel
from global_functions import auto_str_repr


@auto_str_repr
class VizualizeParams():

    def __init__(self, start_epoch: int, end_epoch: int, tick_space: int, plot_dots: bool, plot_text: bool,
                 metrics: Union[None, List[str]]):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.tick_space = tick_space
        self.plot_dots = plot_dots
        self.plot_text = plot_text
        self.metrics = metrics


@auto_str_repr
class PlotParams():
    def __init__(self, font_size: int, figure_size: Tuple[Union[float, int], Union[float, int]], show_title: bool):
        self.font_size = font_size
        self.figure_size = figure_size
        self.show_title = show_title


class Vizualizer():

    @staticmethod
    def myceil(x, base):
        import math
        return base * math.ceil(x / base)

    @staticmethod
    def vizualize_history(train_history_model: TrainHistoryModel, vizualizeParams: VizualizeParams):
        print('Model name: ', train_history_model.model_name)
        print('Stopped epoch: ', train_history_model.stopped_epoch)
        print('Best epoch: ', train_history_model.best_epoch)
        print('Value to monitor: ', train_history_model.value_to_monitor)
        print('Monitor best value: ', train_history_model.monitor_best_value)
        epochs = train_history_model.stopped_epoch
        lines = []
        plt.figure(figsize=(10, 10))

        start_index = max(min(vizualizeParams.start_epoch - 1, epochs), 0)
        end_index = min(vizualizeParams.end_epoch, epochs)

        for metric, values in train_history_model.values.items():
            # if next((s for s in metrics if metric in s.lower()), None) is not None:
            if vizualizeParams.metrics is not None:
                if next((s for s in vizualizeParams.metrics if s.lower() in metric.lower()), None) is None:
                    continue
            x = range(start_index + 1, end_index + 1)
            y = values[start_index:end_index]

            x, y, metric = Vizualizer.edit_values_before_plotting(x, y, metric)

            if vizualizeParams.plot_dots:
                plt.plot(x, y, 'ro', label=metric)
            line, = plt.plot(x, y, label=metric)

            if vizualizeParams.plot_text:
                for i_x, i_y in zip(x, y):
                    plt.text(i_x, i_y, '{0:.4f}'.format(round(i_y, 4)))
            lines.append(line)
            print('metric: ' + metric)
            print("Values: ", str(values).replace('.', ','))

        plt.legend(handles=lines)

        plt.xticks(range(Vizualizer.myceil(start_index + 1, vizualizeParams.tick_space), end_index + 1,
                         vizualizeParams.tick_space))
        # plt.xticks(range(10, 20, vizualizeParams.tick_space))

        plt.xlabel('Epoch')
        plt.title(train_history_model.model_name)
        plt.show()

    @staticmethod
    def edit_values_before_plotting(original_x, original_y, original_metric_name: str):
        return original_x, original_y, original_metric_name
        # if not original_metric_name.endswith(constants.Metrics.ACCURACY):
        #     return original_x, original_y, original_metric_name
        # else:
        #     new_metric_name = original_metric_name[:-len(constants.Metrics.ACCURACY)] + 'error_rate'
        #     new_y = [1.0 - y for y in original_y]
        #     return original_x, new_y, new_metric_name

    @staticmethod
    def vizualize(train_history: List[TrainHistory], vizualizeParams: VizualizeParams):
        for model_index, train_history in enumerate(train_history):
            print('Experiment type', train_history.experiment_type)
            print('Model index: ', model_index + 1)
            print("Parameters: ", train_history.train_parameters)
            for train_history_concrete in train_history.train_history_model_list:
                Vizualizer.vizualize_history(train_history_concrete, vizualizeParams)

    @staticmethod
    def show_random_autoencoder_images(dataset: Dataset, test_images, predictions, number_of_images,
                                       plot_params: PlotParams, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indexes = np.random.randint(test_images.shape[0], size=number_of_images)
        Vizualizer.show_autoencoder_images(dataset, test_images, predictions, indexes, plot_params)

    @staticmethod
    def show_autoencoder_images(dataset: Dataset, test_images, predictions, images_indexes, plot_params: PlotParams):
        font = {'weight': 'bold',
                'size': plot_params.font_size}
        plt.rc('font', **font)
        number_of_images = len(images_indexes)
        plt.figure(figsize=plot_params.figure_size)
        for i, image_idx in enumerate(images_indexes):
            # plot original image
            ax = plt.subplot(2, number_of_images, i + 1)
            if plot_params.show_title:
                plt.title("Originál")
            plt.imshow(dataset.process_image_for_plotting(test_images[image_idx]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, number_of_images, number_of_images + i + 1)
            plt.imshow(dataset.process_image_for_plotting(predictions[image_idx]))  # reshape(28, 28)
            if plot_params.show_title:
                plt.title("Rekonštruovaný")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        # plt.tight_layout(True)
        plt.show()

    @staticmethod
    def show_random_images(dataset: Dataset, images, nrows: int, ncols: int, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indexes = np.random.randint(images.shape[0], size=nrows * ncols)

        fig = plt.figure(figsize=(16, 16))

        for i in range(0, ncols * nrows):
            img = dataset.process_image_for_plotting(images[indexes[i]])
            fig.add_subplot(nrows, ncols, i + 1)
            plt.axis('off')
            plt.imshow(img)

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
