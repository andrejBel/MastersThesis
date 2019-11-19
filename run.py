import vizualizer, experiments , constants, importlib, datasets, models, predictors, training_data
import numpy as np


results = experiments.ExperimentBase.load_experiment_results(constants.ExperimentsPaths.FashionMnist.AUTOENCODER, True)
experiment = results[1].experiment
trained_models = experiment.model_provider()
#trained_models = experiment.model_provider()
#print(len(results))
#vizualizer.Vizualizer.vizualize([result.train_history for result in results], 0.5)
fashion_mnist = datasets.FashionMnistDataset()
predictions_autoencoder = experiments.ExperimentAutoencoder.predict_test(fashion_mnist ,trained_models)
#experiments.ExperimentAutoencoder.evaluate_on_test(fashion_mnist, trained_models)
vizualizer.Vizualizer.show_random_autoencoder_images(fashion_mnist, fashion_mnist.get_test_images(), predictions_autoencoder, 5, 5)