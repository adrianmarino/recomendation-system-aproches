import numpy as np
import tensorflow

"""
from kerastuner.engine.hyperparameters import HyperParameters


def plot_default_hyper_model(
        hyper_model,
        show_shapes=True,
        show_layer_names=True,
        dpi="100",
        rankdir="LR"
):
    model = hyper_model.build(HyperParameters())
    return tensorflow.keras.utils.plot_model(
        model,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        dpi=dpi,
        rankdir=rankdir
    )
"""


def random_color():
    return np.random.rand(3,)


def constains(array, element, comparator=lambda x, y: np.array_equal(x, y)):
    return len([e for e in array if comparator(e, element)]) > 0


def random_colors(count, distance=0.4):
    colors = list()
    comparator = lambda x, y: np.linalg.norm(x-y) < distance
    while len(colors) < count:
        color = random_color()
        if not constains(colors, color, comparator):
            colors.append(color)
    return colors


def plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        dpi="60",
        rankdir="LR"
):
    return tensorflow.keras.utils.plot_model(
        model,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        dpi=dpi,
        rankdir=rankdir
    )
