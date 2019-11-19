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


class Datasets(Const):
    MNIST = 'MNIST'
    FASHION_MNIST = 'FASHION_MNIST'
    CIFAR_10 = 'CIFAR_10'


class GpuConfig(Const):
    DEFAULT, LIMIT, CONTINOUS_GROWTH = range(3)


class Models(Const):
    AUTOENCODER = 'autoencoder'
    CLASSIFIER = 'classifier'
    AUTO_CLASSIFIER = 'auto_classifier'
    ENCODED_OUT = 'ENCODED_OUT'
    DECODED_OUT = 'DECODED_OUT'
    CLASSIFIER_OUT = 'CLASSIFIER_OUT'
    INPUT_SHAPE = 'INPUT_SHAPE'


class ExperimentsPaths(Const):
    class FashionMnist:
        FASHION_MNIST_POSTFIX = '_fm.json'
        AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + FASHION_MNIST_POSTFIX  # 1
        CLASSIFIER = Paths.OUTPUT_DIRECTORY + 'classifier' + FASHION_MNIST_POSTFIX  # 2
        AUTOENCODER1_CLASSIFIER1_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_off' + FASHION_MNIST_POSTFIX  # 3
        AUTOENCODER1_CLASSIFIER1_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder1_classifier1_auto_l_on' + FASHION_MNIST_POSTFIX  # 4
        AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_OFF = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_off' + FASHION_MNIST_POSTFIX  # 5
        AUTOENCODER_TRAINED_CLASSIFIER_AUTO_L_ON = Paths.OUTPUT_DIRECTORY + 'autoencoder_trained_classifier_auto_l_on' + FASHION_MNIST_POSTFIX  # 6
        AUTO_CLASSIFIER = Paths.OUTPUT_DIRECTORY + 'autoclassifier' + FASHION_MNIST_POSTFIX #7
        CLASSIFIER_LAYERS_OFF = Paths.OUTPUT_DIRECTORY + 'classifier_layers_off' + FASHION_MNIST_POSTFIX #8

    class Mnist:
        MNIST_POSTFIX = '_m.json'
        AUTOENCODER = Paths.OUTPUT_DIRECTORY + 'autoencoder' + MNIST_POSTFIX  # 1
        CLASSIFIER = Paths.OUTPUT_DIRECTORY + 'classifier' + MNIST_POSTFIX  # 2


class Metrics(Const):
    ACCURACY = 'acc'
    LOSS = 'loss'
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'
    VAL_CLASSIFIER_OUT_ACCURACY = 'val_' + Models.CLASSIFIER_OUT + '_accuracy'
    VAL_AUTOENCODER_OUT_LOSS = 'val_' + Models.DECODED_OUT + '_loss'
