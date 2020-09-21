from tensorflow.keras.layers import Embedding, Reshape
from tensorflow.keras.regularizers import l2


class EmbeddingLayer:
    def __init__(self, n_items, n_factors, name='', l2_delta=1e-6, initializer='he_normal'):
        self.n_items = n_items
        self.n_factors = n_factors
        self.l2_delta = l2_delta
        self.initializer = initializer
        self.name = name

    def __call__(self, x):
        x = Embedding(
            self.n_items,
            self.n_factors,
            embeddings_initializer=self.initializer,
            embeddings_regularizer=l2(self.l2_delta),
            name=self.name
        )(x)
        return Reshape(target_shape=(self.n_factors,), name=f'{self.name}_3_to_2_dim')(x)
