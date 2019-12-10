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


class Datasets(Const):
    MNIST = 'mnist'
    MNIST_ABBREV = 'm'
    FASHION_MNIST = 'fashion_mnist'
    FASHION_MNIST_ABBREV = 'fm'
    CIFAR_10 = 'cifar_10'
    CIFAR_10_ABBREV = 'c10'


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
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 8
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 3
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + POSTFIX  # 5
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_on' + POSTFIX  # 6
            AUTO_CLASSIFIER = Paths.OUTPUT_DIRECTORY + 'autoclassifier' + POSTFIX #7

        class BasicModelBuilderWithAveragePoolingWithoutDenseBuilder:
            BASIC_MODEL_BUILDER_WITHOUT_DENSE_POSTFIX = '_bmwawt'
            POSTFIX = BASIC_MODEL_BUILDER_WITHOUT_DENSE_POSTFIX + '_' + Datasets.FASHION_MNIST_ABBREV + Paths.JSON_FILE_FORMAT

            CLASSIFIER_EN_LAYERS_ON = Paths.OUTPUT_DIRECTORY + 'classifier_encoder_layers_on' + POSTFIX  # 2
            CLASSIFIER_EN_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_encode_layers_off' + POSTFIX  # 8
            AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + POSTFIX  # 3
            AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + POSTFIX  # 4
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + POSTFIX  # 5
            AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_on' + POSTFIX  # 6

    class Mnist:
        class BasicModelBuilder:
            BASIC_MODEL_BUILDER_POSTFIX = '_bmp'
            POSTFIX = BASIC_MODEL_BUILDER_POSTFIX + '_' + Datasets.MNIST_ABBREV + Paths.JSON_FILE_FORMAT
            AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + POSTFIX  # 1
            CLASSIFIER = Paths.OUTPUT_DIRECTORY + 'classifier' + POSTFIX  # 2


class Metrics(Const):
    ACCURACY = 'acc'
    LOSS = 'loss'
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'
    VAL_CLASSIFIER_OUT_ACCURACY = 'val_' + Models.CLASSIFIER_OUT + '_accuracy'
    VAL_AUTOENCODER_OUT_LOSS = 'val_' + Models.DECODED_OUT + '_loss'
