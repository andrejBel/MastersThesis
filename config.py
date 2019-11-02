import constants

class Config():
    IS_COLAB = False
    try:
        import google.colab
        IS_COLAB = True
    except:
        pass

    IS_LINUX = False
    from sys import platform
    if platform == "linux" or platform == "linux2":
        IS_LINUX = True

    DISABLE_GPU = False
    if IS_COLAB:
        GPU_CONFIG = constants.GpuConfig.CONTINOUS_GROWTH
    else:
        GPU_CONFIG = constants.GpuConfig.LIMIT
    GPU_LIMIT = 768

