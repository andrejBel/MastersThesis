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

@run_once
def on_start():
    import os
    os.makedirs(Constants.Paths.OUTPUT_DIRECTORY, exist_ok=True)
    if Config.IS_LINUX:
        pass
        # !pip install tensorflow - gpu
        # !pip install jsonpickle


