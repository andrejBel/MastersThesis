import named_constants

class Constants(named_constants.Constants):
    class Datasets(named_constants.Constants):
        MNIST, FASHION_MNIST, CIFAR_10 = range(3)

    class Paths(named_constants.Constants):
        OUTPUT_DIRECTORY = './output_dir/'

    class GpuConfig(named_constants.Constants):
        DEFAULT, LIMIT, CONTINOUS_GROWTH = range(3)

    class Models(named_constants.Constants):
        ENCODED_OUT = 'ENCODED_OUT'
        DECODED_OUT = 'DECODED_OUT'
        CLASSIFIER_OUT = 'CLASSIFIER_OUT'
        INPUT_SHAPE = 'INPUT_SHAPE'

