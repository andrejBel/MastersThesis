from constants import Constants

class Config():
    IS_LINUX = False
    from sys import platform

    if platform == "linux" or platform == "linux2":
        IS_LINUX = True

    GPU_CONFIG = Constants.GpuConfig.LIMIT

print(Config.IS_LINUX)