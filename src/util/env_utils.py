import pandas as pd
import tensorflow as tf
from hurry.filesize import size
from tensorflow.python.client import device_lib


def tf_detected_devices():
    devices = device_lib.list_local_devices()
    return pd.DataFrame(
        [(d.name, d.device_type, size(d.memory_limit), d.physical_device_desc) for d in devices],
        columns=['Name', 'Device Type', 'Memory', 'Description']
    )


def tf_version():
    return f'Tensorflow version: {tf.__version__}'
