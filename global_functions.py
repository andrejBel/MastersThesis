from functools import wraps

import constants
from config import Config


def execute_before(function_to_run_before, *args_f, **kwargs_f):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            function_to_run_before(*args_f, **kwargs_f)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def run_once(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def auto_str_repr(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    cls.__repr__ = __str__
    return cls

@run_once
def on_start():
    import os
    os.makedirs(constants.Paths.OUTPUT_DIRECTORY, exist_ok=True)
    if Config.DISABLE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if Config.IS_COLAB:
        pass
        # !pip install tensorflow - gpu #2.0.0
        # !pip install jsonpickle #1.2
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        if Config.GPU_CONFIG is constants.GpuConfig.CONTINOUS_GROWTH:
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        elif Config.GPU_CONFIG is constants.GpuConfig.LIMIT:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=Config.GPU_LIMIT)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    if tf.test.is_gpu_available():
        print("GPU OK")
    else:
        print("GPU NOT FOUND")


def coalesce(value, default):
    return value if value is not None else default


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses
