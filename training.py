import constants
import datasets
import experiments
import models
import training_data
from global_functions import on_start, execute_before


class TrainingMnist():
    # 1
    @staticmethod
    @execute_before(on_start)
    def autoencoder():
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoder(datasets.MnistDataset,
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
    def classifier(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentClassifier(datasets.MnistDataset,
                                                   models.BasicModelBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6,
                                                                            autoencoder_layers_trainable_during_classification_training,
                                                                            constants.ExperimentsPaths.Mnist.BasicModelBuilder.CLASSIFIER_EN_LAYERS_ON
                                                                            if autoencoder_layers_trainable_during_classification_training else
                                                                            constants.ExperimentsPaths.Mnist.BasicModelBuilder.CLASSIFIER_EN_LAYERS_OFF,
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
                datasets.MnistDataset,
                models.BasicModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     autoencoder_layers_trainable_during_classification_training,
                                                                     constants.ExperimentsPaths.Mnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON
                                                                     if autoencoder_layers_trainable_during_classification_training else
                                                                     constants.ExperimentsPaths.Mnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF,
                                                                     True)
            result = exp.train(parameters)
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
                                                    models.LargeModelBuilder.get_provider(),
                                                    training_data.BasicTrainingDataGeneratorAutoencoder())
            result = exp.train(training_data.BasicTrainParametersAutoencoder(100, 128, 20, True, rate, 1e-5,
                                                                             constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTOENCODER,
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
                                                   models.LargeModelBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6,
                                                                            autoencoder_layers_trainable_during_classification_training,
                                                                            constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.CLASSIFIER_EN_LAYERS_ON
                                                                            if autoencoder_layers_trainable_during_classification_training else
                                                                            constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.CLASSIFIER_EN_LAYERS_OFF,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for classifier_rate in [0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.FashionMnistDataset,
                models.LargeModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     autoencoder_layers_trainable_during_classification_training,
                                                                     constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON
                                                                     if autoencoder_layers_trainable_during_classification_training else
                                                                     constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF,
                                                                     True)
            result = exp.train(parameters)
            experiment_results.append(result)
        return experiment_results

    # 5, 6
    @staticmethod
    @execute_before(on_start)
    def autoencoder_trained_then_classifier(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        rates = []
        if autoencoder_layers_trainable_during_classification_training:
            rates = [0.01, 0.05, 0.1, 0.5, 1]
        else:
            rates = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
        for rate in rates:
            model_provider = experiments.ExperimentBase.provide_existing_model_from_log(
                constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTOENCODER,
                experiments.ExperimentAutoencoder.predicate_for_choosing_best_model())
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDataset,
                                                   model_provider,
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(
                60, 128, 20, True, rate, 1e-6,
                autoencoder_layers_trainable_during_classification_training,
                constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON
                if autoencoder_layers_trainable_during_classification_training else
                constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF,
                True)
            )
            experiment_results.append(result)
        return experiment_results

    # 8
    @staticmethod
    @execute_before(on_start)
    def auto_classifier():
        experiment_results = []
        # for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
        for rate in [1]:
            exp = experiments.ExperimentAutoClassifier(datasets.FashionMnistDataset,
                                                       models.LargeModelBuilder.get_provider(),
                                                       training_data.BasicTrainingDataGeneratorAutoClassifier())
            result = exp.train(
                training_data.BasicTrainParametersAutoClassifier(100, 128, 20, True, rate, 1e-6, 1.0, 2.0,
                                                                 constants.ExperimentsPaths.FashionMnist.LargeModelBuilder.AUTO_CLASSIFIER_1to2,
                                                                 True))
            experiment_results.append(result)
        return experiment_results

    @staticmethod
    def run_all():
        TrainingFashionMnist.autoencoder()
        TrainingFashionMnist.classifier(True)
        TrainingFashionMnist.classifier(False)
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
                datasets.FashionMnistDataset,
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


class TrainingCifarGrayBasicModelBuilder():

    # 1
    @staticmethod
    @execute_before(on_start)
    def autoencoder():
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoder(datasets.Cifar10GrayDataset,
                                                    models.BasicModelBuilder.get_provider(),
                                                    training_data.BasicTrainingDataGeneratorAutoencoder())
            result = exp.train(training_data.BasicTrainParametersAutoencoder(100, 128, 20, True, rate, 1e-5,
                                                                             constants.ExperimentsPaths.CifarGray.BasicModelBuilder.AUTOENCODER,
                                                                             True))
            experiment_results.append(result)
        return experiment_results

    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentClassifier(datasets.Cifar10GrayDataset,
                                                   models.BasicModelBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6,
                                                                            autoencoder_layers_trainable_during_classification_training,
                                                                            constants.ExperimentsPaths.CifarGray.BasicModelBuilder.CLASSIFIER_EN_LAYERS_ON
                                                                            if autoencoder_layers_trainable_during_classification_training else
                                                                            constants.ExperimentsPaths.CifarGray.BasicModelBuilder.CLASSIFIER_EN_LAYERS_OFF,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for classifier_rate in [1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.Cifar10GrayDataset,
                models.BasicModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     autoencoder_layers_trainable_during_classification_training,
                                                                     constants.ExperimentsPaths.CifarGray.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON
                                                                     if autoencoder_layers_trainable_during_classification_training else
                                                                     constants.ExperimentsPaths.CifarGray.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF,
                                                                     True)
            result = exp.train(parameters)
            experiment_results.append(result)
        return experiment_results


class TrainingCifarGrayLargeModelBuilder():

    # 1
    @staticmethod
    @execute_before(on_start)
    def autoencoder():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentAutoencoder(datasets.Cifar10GrayDataset,
                                                    models.LargeModelBuilder.get_provider(),
                                                    training_data.BasicTrainingDataGeneratorAutoencoder())
            result = exp.train(training_data.BasicTrainParametersAutoencoder(100, 128, 20, True, rate, 1e-5,
                                                                             constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER,
                                                                             True))
            experiment_results.append(result)
        return experiment_results

    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier_layers_on():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentClassifierWithLearningRate(datasets.Cifar10GrayDataset,
                                                                   models.LargeModelBuilder.get_provider(),
                                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(150, 128, 50, True, rate, 1e-6, True,
                                                                            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.CLASSIFIER_EN_LAYERS_ON,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together_layers_on():
        experiment_results = []
        for classifier_rate in [1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.Cifar10GrayDataset,
                models.LargeModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     True,
                                                                     constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON,
                                                                     True)
            result = exp.train(parameters)
            experiment_results.append(result)
        return experiment_results

    # 5, 6
    @staticmethod
    @execute_before(on_start)
    def autoencoder_trained_then_classifier_encoder_layers_off():
        for rate in [1]:
            model_provider = experiments.ExperimentBase.provide_existing_model_from_log(
                constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER,
                experiments.ExperimentAutoencoder.predicate_for_choosing_best_model())
            autoencoder_trained_model = model_provider()
            model_with_classifier = models.LargeModelBuilder.get_provider()(
                datasets.Cifar10GrayDataset())
            model_with_classifier.autoencoder.set_weights(autoencoder_trained_model.autoencoder.get_weights())
            result_model = models.ExistingModelProvider(model_with_classifier)
            exp = experiments.ExperimentClassifier(datasets.Cifar10GrayDataset,
                                                   result_model,
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(
                100, 128, 20, True, rate, 1e-6,
                False,
                constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF,
                True))

    # 7
    @staticmethod
    @execute_before(on_start)
    def auto_classifier1to1():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentAutoClassifier(datasets.Cifar10GrayDataset,
                                                       models.LargeModelBuilder.get_provider(),
                                                       training_data.BasicTrainingDataGeneratorAutoClassifier())
            result = exp.train(
                training_data.BasicTrainParametersAutoClassifier(100, 128, 20, True, rate, 1e-6, 1.0, 1.0,
                                                                 constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTO_CLASSIFIER_1to1,
                                                                 True))
            experiment_results.append(result)
        return experiment_results

    @staticmethod
    @execute_before(on_start)
    def auto_classifier1to5():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentAutoClassifier(datasets.Cifar10GrayDataset,
                                                       models.LargeModelBuilder.get_provider(),
                                                       training_data.BasicTrainingDataGeneratorAutoClassifier())
            result = exp.train(
                training_data.BasicTrainParametersAutoClassifier(150, 128, 20, True, rate, 1e-6, 1.0, 5.0,
                                                                 constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTO_CLASSIFIER_1to5,
                                                                 True))
            experiment_results.append(result)
        return experiment_results

    @staticmethod
    @execute_before(on_start)
    def auto_classifier5to1():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentAutoClassifier(datasets.Cifar10GrayDataset,
                                                       models.LargeModelBuilder.get_provider(),
                                                       training_data.BasicTrainingDataGeneratorAutoClassifier())
            result = exp.train(
                training_data.BasicTrainParametersAutoClassifier(150, 128, 20, True, rate, 1e-6, 5.0, 1.0,
                                                                 constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTO_CLASSIFIER_5to1,
                                                                 True))
            experiment_results.append(result)
        return experiment_results


class TrainingCifarGrayFiveLargeModelBuilder():

    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier_layers_on():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentClassifierWithLearningRate(datasets.Cifar10GrayFiveDataset,
                                                                   models.LargeModelBuilder.get_provider(),
                                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(170, 128, 50, True, rate, 1e-6, True,
                                                                            constants.ExperimentsPaths.CifarGrayFiveClasses.LargeModelBuilder.CLASSIFIER_EN_LAYERS_ON,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together_layers_on():
        experiment_results = []
        for classifier_rate in [1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.Cifar10GrayFiveDataset,
                models.LargeModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     True,
                                                                     constants.ExperimentsPaths.CifarGrayFiveClasses.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON,
                                                                     True)
            result = exp.train(parameters)
            experiment_results.append(result)
        return experiment_results

    @staticmethod
    @execute_before(on_start)
    def auto_classifier1to1():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentAutoClassifier(datasets.Cifar10GrayFiveDataset,
                                                       models.LargeModelBuilder.get_provider(),
                                                       training_data.BasicTrainingDataGeneratorAutoClassifier())
            result = exp.train(
                training_data.BasicTrainParametersAutoClassifier(100, 128, 20, True, rate, 1e-6, 1.0, 1.0,
                                                                 constants.ExperimentsPaths.CifarGrayFiveClasses.LargeModelBuilder.AUTO_CLASSIFIER_1to1,
                                                                 True))
            experiment_results.append(result)
        return experiment_results


class TrainingFashionMnistFive():

    # 1
    @staticmethod
    @execute_before(on_start)
    def autoencoder():
        experiment_results = []
        for rate in [1]:
            exp = experiments.ExperimentAutoencoder(datasets.FashionMnistDatasetFiveClasses,
                                                    models.BasicModelBuilder.get_provider(),
                                                    training_data.BasicTrainingDataGeneratorAutoencoder())
            result = exp.train(training_data.BasicTrainParametersAutoencoder(100, 128, 20, True, rate, 1e-5,
                                                                             constants.ExperimentsPaths.FashionMnistDatasetFiveClasses.BasicModelBuilder.AUTOENCODER,
                                                                             True))
            experiment_results.append(result)
        return experiment_results

    # 2
    @staticmethod
    @execute_before(on_start)
    def classifier(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for rate in [0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentClassifier(datasets.FashionMnistDatasetFiveClasses,
                                                   models.BasicModelBuilder.get_provider(),
                                                   training_data.BasicTrainingDataGeneratorClassifier())
            result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, rate, 1e-6,
                                                                            autoencoder_layers_trainable_during_classification_training,
                                                                            constants.ExperimentsPaths.FashionMnistDatasetFiveClasses.BasicModelBuilder.CLASSIFIER_EN_LAYERS_ON
                                                                            if autoencoder_layers_trainable_during_classification_training else
                                                                            constants.ExperimentsPaths.FashionMnistDatasetFiveClasses.BasicModelBuilder.CLASSIFIER_EN_LAYERS_OFF,
                                                                            True)
                               )
            experiment_results.append(result)
        return experiment_results

    # 3, 4
    @staticmethod
    @execute_before(on_start)
    def autoencoder_classifier_together(autoencoder_layers_trainable_during_classification_training: bool):
        experiment_results = []
        for classifier_rate in [0.1, 0.25, 0.5, 1]:
            exp = experiments.ExperimentAutoencoderAndClassifier(
                datasets.FashionMnistDatasetFiveClasses,
                models.BasicModelBuilder.get_provider(),
                training_data.BasicTrainingDataGeneratorAutoencoderClassifier())
            parameters_autoencoder = training_data.TrainParameters(1, 128, None, False, 1.0, 1e-7)
            parameters_classifier = training_data.TrainParameters(1, 128, None, True, classifier_rate, 1e-7)
            parameters = training_data.BasicTrainParametersTwoModels(parameters_autoencoder, parameters_classifier, 150,
                                                                     autoencoder_layers_trainable_during_classification_training,
                                                                     constants.ExperimentsPaths.FashionMnistDatasetFiveClasses.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON
                                                                     if autoencoder_layers_trainable_during_classification_training else
                                                                     constants.ExperimentsPaths.FashionMnistDatasetFiveClasses.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF,
                                                                     True)
            result = exp.train(parameters)
            experiment_results.append(result)
        return experiment_results


class TrainingDerivatedDatasets:

    # 2
    @staticmethod
    @execute_before(on_start)
    def train_derivated_and_merged_dataset(path_to_autoencoder: str,
                                           originalDataset: datasets.Dataset,
                                           model_provider: models.ModelProviderBase,
                                           path_to_save_trained_derivated_model: str,
                                           path_to_save_trained_merged_model: str):
        results = experiments.ExperimentBase.load_experiment_results(
            path_to_autoencoder, True)
        try:
            results.sort(key=lambda result: result.train_history.train_parameters.train_data_rate)
        except:
            results.sort(
                key=lambda result: result.train_history.train_parameters.train_parameters_classifier.train_data_rate)

        experiment = results[-1].experiment
        trained_models_autoencoder = experiment.model_provider()

        newDataset = datasets.DerivatedDataset(originalDataset, trained_models_autoencoder)

        exp = experiments.ExperimentClassifier(newDataset.provide,
                                               model_provider,
                                               training_data.BasicTrainingDataGeneratorClassifier())
        result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, 1, 1e-6,
                                                                        True,
                                                                        path_to_save_trained_derivated_model,
                                                                        True)
                           )
        new_trained_classifier, _ = result

        merged_dataset = datasets.MergedDataset(originalDataset, newDataset)
        exp = experiments.ExperimentClassifier(merged_dataset.provide,
                                               model_provider,
                                               training_data.BasicTrainingDataGeneratorClassifier())
        result = exp.train(training_data.BasicTrainParametersClassifier(100, 128, 20, True, 1, 1e-6,
                                                                        True,
                                                                        path_to_save_trained_merged_model,
                                                                        True)
                           )
        new_trained_merged_classifier, _ = result
        return new_trained_classifier, new_trained_merged_classifier

    @staticmethod
    def train_fashion_mnist():
        print("nove 2")
        # TrainingDerivatedDatasets.train_derivated_and_merged_dataset(
        #     constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON,
        #     datasets.FashionMnistDataset(),
        #     models.BasicModelBuilder.get_provider(),
        #     constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_DERIVATED_CLASSIFIER,
        #     constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_MERGED_CLASSIFIER
        # )
        TrainingDerivatedDatasets.train_derivated_and_merged_dataset(
            constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTO_CLASSIFIER_1to1,
            datasets.FashionMnistDataset(),
            models.BasicModelBuilder.get_provider(),
            constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTO_CLASSIFIER_1to1_DERIVATED_CLASSIFIER,
            constants.ExperimentsPaths.FashionMnist.BasicModelBuilder.AUTO_CLASSIFIER_1to1_MERGED_CLASSIFIER
        )

    @staticmethod
    def train_cifar():
        TrainingDerivatedDatasets.train_derivated_and_merged_dataset(
            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON,
            datasets.Cifar10GrayDataset(),
            models.LargeModelBuilder.get_provider(),
            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_DERIVATED_CLASSIFIER,
            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_MERGED_CLASSIFIER
        )
        TrainingDerivatedDatasets.train_derivated_and_merged_dataset(
            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTO_CLASSIFIER_1to1,
            datasets.Cifar10GrayDataset(),
            models.LargeModelBuilder.get_provider(),
            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTO_CLASSIFIER_1to1_DERIVATED_CLASSIFIER,
            constants.ExperimentsPaths.CifarGray.LargeModelBuilder.AUTO_CLASSIFIER_1to1_MERGED_CLASSIFIER
        )
