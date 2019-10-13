import named_constants


class Constants(named_constants.Constants):
    class Datasets(named_constants.Constants):
        MNIST, FASHION_MNIST, CIFAR_10 = range(3)

    class Paths(named_constants.Constants):
        OUTPUT_DIRECTORY = './output_dir/'

    class GpuConfig(named_constants.Constants):
        DEFAULT, LIMIT, CONTINOUS_GROWTH = range(3)



print(Constants.Paths.OUTPUT_DIRECTORY)