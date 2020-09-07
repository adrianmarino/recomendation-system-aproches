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


def plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        dpi="100",
        rankdir="LR"
):
    return tensorflow.keras.utils.plot_model(
        model,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        dpi=dpi,
        rankdir=rankdir
    )
