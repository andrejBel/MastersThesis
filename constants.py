import named_constants

class MetaConst(type):
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise TypeError

class Const(metaclass=MetaConst):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        raise TypeError

class Constants(Const):
    class Datasets(Const):
        MNIST = 'MNIST'
        FASHION_MNIST = 'FASHION_MNIST'
        CIFAR_10 = 'CIFAR_10'

    class Paths(Const):
        OUTPUT_DIRECTORY = './output_dir/'

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

    class Experiments(Const):
        PATIENCE = 20
        AUTOENCODER = 'AUTOENCODER'
        CLASSIFIER = 'CLASSIFIER'
        AE_CL_AU_L_OFF = 'AUTOENCODER_CLASSIFIER_AUTOENCODER_LAYERS_OFF'
        AE_CL_AU_L_ON = 'AUTOENCODER_CLASSIFIER_AUTOENCODER_LAYERS_ON'

    class Metrics(Const):
        ACCURACY = 'acc'
        LOSS = 'loss'
        VAL_LOSS = 'val_loss'
        VAL_ACCURACY = 'val_accuracy'

    class Logger(Const):
        ROOT_KEY = 'modellist'