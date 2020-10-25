from tensorflow.python.keras import Input

from model.embedding.embedding_layer import EmbeddingLayer


def create_embedding_input(n_items, n_min_factors, name, shape=(1,)):
    input = Input(shape=shape, name=f'{name}_idx')

    return EmbeddingLayer(
        n_items=n_items,
        n_min_factors=n_min_factors,
        name=f'{name}_embedding')(input)

