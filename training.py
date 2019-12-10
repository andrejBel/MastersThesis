import experiments, datasets, training_data
import models
import constants

from global_functions import on_start, execute_before


class TrainingMnist():
    # 1
    @staticmethod
    @execute_before(on_start)
    def autoencoder():
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoder(datasets.FashionMnistDataset,
                                                    models.BasicModelBuilder.get_provider(),
                                                    training_data.BasicTrainingDataGeneratorAutoencoder())
            result = exp.train(training_data.BasicTrainParametersAutoencoder(100, 128, 20, True, rate, 1e-5,
                                                                             constants.ExperimentsPaths.Mnist.BasicModelBuilder.AUTOENCODER,
                                                                             True))
            experiment_results.append(result)
        return experiment_results

    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier():
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDataset,
                                                   models.BasicModelBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6, True,
                                                                            constants.ExperimentsPaths.Mnist.BasicModelBuilder.CLASSIFIER,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results


class TrainingFashionMnist():

    # 1
    @staticmethod
    @execute_before(on_start)
    def autoencoder():
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoder(datasets.FashionMnistDataset,
                                                    models.BasicModelBuilder.get_provider(),
                                                    training_data.BasicTrainingDataGeneratorAutoencoder())
            result = exp.train(training_data.BasicTrainParametersAutoencoder(100, 128, 20, True, rate, 1e-5,
                                                                             constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER,
                                                                             True))
            experiment_results.append(result)
        return experiment_results

    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDataset,
                                                   models.BasicModelBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6,
                                                                            autoencoder_layers_trainable_during_classification_training,
                                                                            constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.CLASSIFIER_EN_LAYERS_ON
                                                                            if autoencoder_layers_trainable_during_classification_training else
                                                                            constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.CLASSIFIER_EN_LAYERS_OFF,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for classifier_rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.DatasetProviderClass(datasets.FashionMnistDataset),
                models.BasicModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     autoencoder_layers_trainable_during_classification_training,
                                                                     constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON
                                                                     if autoencoder_layers_trainable_during_classification_training else
                                                                     constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF,
                                                                     True)
            result = exp.train(parameters)
            experiment_results.append(result)
        return experiment_results

    # 5, 6
    @staticmethod
    @execute_before(on_start)
    def autoencoder_trained_then_classifier(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            model_provider = experiments.ExperimentBase.provide_existing_model_from_log(
                constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER,
                experiments.ExperimentAutoencoder.predicate_for_choosing_best_model())
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDataset,
                                                   model_provider,
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(
                60, 128, 20, True, rate, 1e-6,
                autoencoder_layers_trainable_during_classification_training,
                constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON
                if autoencoder_layers_trainable_during_classification_training else
                constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF,
                True)
            )
            experiment_results.append(result)
        return experiment_results

    # 7
    @staticmethod
    @execute_before(on_start)
    def auto_classifier():
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoClassifier(datasets.FashionMnistDataset,
                                                       models.BasicModelBuilder.get_provider(),
                                                       training_data.BasicTrainingDataGeneratorAutoClassifier())
            result = exp.train(
                training_data.BasicTrainParametersAutoClassifier(100, 128, 20, True, rate, 1e-6, 1.0, 2.0,
                                                                 constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTO_CLASSIFIER,
                                                                 True))
            experiment_results.append(result)
        return experiment_results

    @staticmethod
    def run_all():
        TrainingFashionMnist.autoencoder()
        TrainingFashionMnist.classifier()
        TrainingFashionMnist.autoencoder_classifier_together(False)
        TrainingFashionMnist.autoencoder_classifier_together(True)
        TrainingFashionMnist.autoencoder_trained_then_classifier(False)
        TrainingFashionMnist.autoencoder_trained_then_classifier(True)
        TrainingFashionMnist.auto_classifier()


class TrainingFashionMnistAveragePoolingWithoutDense():
    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier(autoencoder_layers_trainable_during_classification_training: bool):
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDataset,
                                                   models.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6,
                                                                            autoencoder_layers_trainable_during_classification_training,
                                                                            constants.ExperimentsPaths.FashionMnist.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.CLASSIFIER_EN_LAYERS_ON
                                                                            if autoencoder_layers_trainable_during_classification_training else
                                                                            constants.ExperimentsPaths.FashionMnist.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.CLASSIFIER_EN_LAYERS_OFF,
                                                                            True)
                               )

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together(autoencoder_layers_trainable_during_classification_training: bool):
        for classifier_rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.DatasetProviderClass(datasets.FashionMnistDataset),
                models.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     autoencoder_layers_trainable_during_classification_training,
                                                                     constants.ExperimentsPaths.FashionMnist.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON
                                                                     if autoencoder_layers_trainable_during_classification_training else
                                                                     constants.ExperimentsPaths.FashionMnist.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF,
                                                                     True)
            result = exp.train(parameters)

    # 5, 6
    @staticmethod
    @execute_before(on_start)
    def autoencoder_trained_then_classifier(autoencoder_layers_trainable_during_classification_training: bool):
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            model_provider = experiments.ExperimentBase.provide_existing_model_from_log(
                constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER,
                experiments.ExperimentAutoencoder.predicate_for_choosing_best_model())
            autoencoder_trained_model = model_provider()
            model_with_classifier = models.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.get_provider()(
                datasets.FashionMnistDataset())
            model_with_classifier.autoencoder.set_weights(autoencoder_trained_model.autoencoder.get_weights())
            result_model = models.ExistingModelProvider(model_with_classifier)
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDataset,
                                                   result_model,
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(
                60, 128, 20, True, rate, 1e-6,
                autoencoder_layers_trainable_during_classification_training,
                constants.ExperimentsPaths.FashionMnist.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON
                if autoencoder_layers_trainable_during_classification_training else
                constants.ExperimentsPaths.FashionMnist.BasicModelBuilderWithAveragePoolingWithoutDenseBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF,
                True)
            )
