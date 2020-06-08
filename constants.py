class MetaConst(type):
    def __getattr__(cls, key):
        raise Exception(key + " not found!")

    def __setattr__(cls, key, value):
        raise TypeError


class Const(metaclass=MetaConst):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        raise TypeError


class Paths(Const):
    OUTPUT_DIRECTORY = './output_dir/'
    JSON_FILE_FORMAT = '.json'
    TXT_FILE_FORMAT = '.txt'


class Datasets(Const):
    MNIST = 'mnist'
    MNIST_ABBREV = 'm'
    FASHION_MNIST = 'fashion_mnist'
    FASHION_MNIST_ABBREV = 'fm'
    FASHION_MNIST_FIVE = 'fashion_mnist_five'
    FASHION_MNIST_FIVE_ABBREV = 'fmf'
    FASHION_MNIST_FIVE_OTHER = 'fashion_mnist_five_other'
    FASHION_MNIST_FIVE_OTHER_ABBREV = 'fmfo'

    CIFAR_10 = 'cifar_10'
    CIFAR_10_ABBREV = 'c10'
    CIFAR_10_FIVE = 'cifar_10_five'
    CIFAR_10_FIVE_ABBREV = 'c10f'
    CIFAR_10_FIVE_OTHER = 'cifar_10_five_other'
    CIFAR_10_FIVE_OTHER_ABBREV = 'c10of'
    TRAIN = 'train'
    TEST = 'test'
    DERIVATED = 'derivated'
    MERGED = 'merged'


class GpuConfig(Const):
    DEFAULT, LIMIT, CONTINOUS_GROWTH = range(3)


class Models(Const):
    AUTOENCODER = 'autoencoder'
    CLASSIFIER = 'classifier'
    AUTO_CLASSIFIER = 'auto_classifier'
    ENCODED_OUT = 'encoded_out'
    DECODED_OUT = 'decoded_out'
    CLASSIFIER_OUT = 'classifier_out'
    INPUT_SHAPE = 'input_shape'


class ExperimentsPaths(Const):
    class FashionMnist:
        class BasicModelBuilder:
            BASIC_MODEL_BUILDER_POSTFIX = '_bmp'
            POSTFIX = BASIC_MODEL_BUILDER_POSTFIX + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 3
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 5
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_on' + POSTFIX  # 6
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + POSTFIX  # 7
            AUTO_CLASSIFIER_1to2 = Paths.OUTPUT_DIRECTORY + 'autoclassifier' + POSTFIX  # 8
            AUTO_CLASSIFIER_1to1 = Paths.OUTPUT_DIRECTORY + 'autoclassifier1to1' + POSTFIX  # 8

            AUTOENCODER_DERIVATED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder_" + 'classifier_encoder_layers_on' + "_" + Datasets.DERIVATED + POSTFIX
            AUTOENCODER_MERGED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder_" + 'classifier_encoder_layers_on' + "_" + Datasets.MERGED + POSTFIX

            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_DERIVATED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder1_classifier1_auto_l_on_" + 'classifier_encoder_layers_on' + "_" + Datasets.DERIVATED + POSTFIX
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_MERGED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder1_classifier1_auto_l_on_" + 'classifier_encoder_layers_on' + "_" + Datasets.MERGED + POSTFIX

            AUTO_CLASSIFIER_1to1_DERIVATED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoclassifier1to1_" + 'classifier_encoder_layers_on' + "_" + Datasets.DERIVATED + POSTFIX
            AUTO_CLASSIFIER_1to1_MERGED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoclassifier1to1_" + 'classifier_encoder_layers_on' + "_" + Datasets.MERGED + POSTFIX

        class BasicModelBuilderWithAveragePoolingWithoutDenseBuilder:
            BASIC_MODEL_BUILDER_WITHOUT_DENSE_POSTFIX = '_bmwawt'
            POSTFIX = BASIC_MODEL_BUILDER_WITHOUT_DENSE_POSTFIX + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.JSON_FILE_FORMAT

            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 8
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 3
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + POSTFIX  # 5
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_on' + POSTFIX  # 6

        class LargeModelBuilder:
            LARGE_MODEL_BUILDER_POSTFIX = '_lmp'
            POSTFIX = LARGE_MODEL_BUILDER_POSTFIX + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 3
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 5
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_on' + POSTFIX  # 6
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + POSTFIX  # 7
            AUTO_CLASSIFIER_1to2 = Paths.OUTPUT_DIRECTORY + 'autoclassifier1to2' + POSTFIX  # 8

        class BasicModelBuilderDecorderWithoutSigmoidActivation:
            BASIC_MODEL_BUILDER_WITHOUT_SIGMOID_POSTFIX = '_bmpwsa'
            POSTFIX = BASIC_MODEL_BUILDER_WITHOUT_SIGMOID_POSTFIX + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1

    class FashionMnistDatasetFiveClasses:
        class BasicModelBuilder:
            BASIC_MODEL_BUILDER_POSTFIX = '_bmp'
            POSTFIX = BASIC_MODEL_BUILDER_POSTFIX + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # nie je
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 3 nie je
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4

    class Mnist:
        class BasicModelBuilder:
            BASIC_MODEL_BUILDER_POSTFIX = '_bmp'
            POSTFIX = BASIC_MODEL_BUILDER_POSTFIX + '_' + Datasets.MNIST_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 3 nie je
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 8

    class CifarGray:
        class BasicModelBuilder:
            BASIC_MODEL_BUILDER_POSTFIX = '_bmp'
            POSTFIX = BASIC_MODEL_BUILDER_POSTFIX + '_' + Datasets.CIFAR_10_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 3
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 5 nie je

        class LargeModelBuilder:
            LARGE_MODEL_BUILDER_POSTFIX = '_lmp'
            POSTFIX = LARGE_MODEL_BUILDER_POSTFIX + '_' + Datasets.CIFAR_10_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + POSTFIX  # 7
            AUTO_CLASSIFIER_1to1 = Paths.OUTPUT_DIRECTORY + 'autoclassifier1to1' + POSTFIX  # 8
            AUTO_CLASSIFIER_1to5 = Paths.OUTPUT_DIRECTORY + 'autoclassifier1to5' + POSTFIX  # 8
            AUTO_CLASSIFIER_5to1 = Paths.OUTPUT_DIRECTORY + 'autoclassifier5to1' + POSTFIX  # 8

            AUTOENCODER_DERIVATED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder_" + 'classifier_encoder_layers_on' + "_" + Datasets.DERIVATED + POSTFIX
            AUTOENCODER_MERGED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder_" + 'classifier_encoder_layers_on' + "_" + Datasets.MERGED + POSTFIX

            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_DERIVATED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder1_classifier1_auto_l_on_" + 'classifier_encoder_layers_on' + "_" + Datasets.DERIVATED + POSTFIX
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON_MERGED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoencoder1_classifier1_auto_l_on_" + 'classifier_encoder_layers_on' + "_" + Datasets.MERGED + POSTFIX

            AUTO_CLASSIFIER_1to1_DERIVATED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoclassifier1to1_" + 'classifier_encoder_layers_on' + "_" + Datasets.DERIVATED + POSTFIX
            AUTO_CLASSIFIER_1to1_MERGED_CLASSIFIER = Paths.OUTPUT_DIRECTORY + "autoclassifier1to1_" + 'classifier_encoder_layers_on' + "_" + Datasets.MERGED + POSTFIX

    class CifarGrayFiveClasses:
        class LargeModelBuilder:
            LARGE_MODEL_BUILDER_POSTFIX = '_lmp'
            POSTFIX = LARGE_MODEL_BUILDER_POSTFIX + '_' + Datasets.CIFAR_10_FIVE_ABBREV + Paths.JSON_FILE_FORMAT
            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTO_CLASSIFIER_1to1 = Paths.OUTPUT_DIRECTORY + 'autoclassifier1to1' + POSTFIX  # 8


class AutoencoderEvaluationPaths:
    class FashionMnist:
        class BasicModelBuilder:
            class Train:
                DATASET_TYPE = Datasets.TRAIN
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.FASHION_MNIST_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

            class Test:
                DATASET_TYPE = Datasets.TEST
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.FASHION_MNIST_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

    class FashionMnistDatasetFiveClasses:
        class BasicModelBuilder:
            class Train:
                DATASET_TYPE = Datasets.TRAIN
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.FASHION_MNIST_FIVE_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_OTHER_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_OTHER_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

            class Test:
                DATASET_TYPE = Datasets.TEST
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.FASHION_MNIST_FIVE_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_OTHER_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_OTHER_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

    class Mnist:
        class BasicModelBuilder:
            class Train:
                DATASET_TYPE = Datasets.TRAIN
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.MNIST_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

            class Test:
                DATASET_TYPE = Datasets.TEST
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.MNIST_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

    class CifarGray:
        class BasicModelBuilder:
            class Train:
                DATASET_TYPE = Datasets.TRAIN
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.CIFAR_10_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

            class Test:
                DATASET_TYPE = Datasets.TEST
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.CIFAR_10_ABBREV
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

        class LargeModelBuilder:
            class Train:
                DATASET_TYPE = Datasets.TRAIN
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.CIFAR_10_ABBREV + "_lmb_"
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT

            class Test:
                DATASET_TYPE = Datasets.TEST
                AUTOENCODER_MODEL_IDENTIFIER = Datasets.CIFAR_10_ABBREV + "_lmb_"
                FASHION_MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                FASHION_MNIST_FIVE_CLASSES_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.FASHION_MNIST_FIVE_ABBREV + Paths.TXT_FILE_FORMAT
                MNIST_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.MNIST_ABBREV + Paths.TXT_FILE_FORMAT
                CIFAR_GRAY_EVAL = Paths.OUTPUT_DIRECTORY + DATASET_TYPE + '_' + AUTOENCODER_MODEL_IDENTIFIER + '_' + Datasets.CIFAR_10_ABBREV + Paths.TXT_FILE_FORMAT


class Metrics(Const):
    ACCURACY = 'accuracy'
    LOSS = 'loss'
    CLASSIFIER_OUT_ACCURACY = Models.CLASSIFIER_OUT + '_accuracy'
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'
    VAL_CLASSIFIER_OUT_ACCURACY = 'val_' + Models.CLASSIFIER_OUT + '_accuracy'
    VAL_AUTOENCODER_OUT_LOSS = 'val_' + Models.DECODED_OUT + '_loss'
