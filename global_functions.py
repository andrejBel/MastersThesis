from functools import wraps
from constants import Constants
from config import Config

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
    cls.__repr__ =  __str__
    return cls


def bind_method_to_instance(instance, method):
    def binding_scope_fn(*args, **kwargs):
        return method(instance, *args, **kwargs)
    return binding_scope_fn

@run_once
def on_start():
    import os
    os.makedirs(Constants.Paths.OUTPUT_DIRECTORY, exist_ok=True)
    if Config.IS_LINUX:
        pass
        # !pip install tensorflow - gpu
        # !pip install jsonpickle
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    import tensorflow as tf
    if gpus:
        if Config.GPU_CONFIG is Constants.GpuConfig.CONTINOUS_GROWTH:
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        elif Config.GPU_CONFIG is Constants.GpuConfig.LIMIT:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    if tf.test.is_gpu_available():
        print("GPU OK")
    else:
        print("GPU NOT FOUND")