from tensorflow.keras.layers import Input, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import EmbeddingLayer


class EmbeddingDotModelFactory:
    @staticmethod
    def create(
            n_users,
            n_movies,
            n_factors,
            lr=0.001,
            l2_delta=1e-6,
            loss='mean_squared_error'
    ):
        user_input = Input(shape=(1,), name='users_idx')
        movie_input = Input(shape=(1,), name='movies_idx')

        user_emb = EmbeddingLayer(n_users, n_factors, l2_delta=l2_delta, name='users_embedding')(user_input)
        movie_emb = EmbeddingLayer(n_movies, n_factors, l2_delta=l2_delta, name='movies_embedding')(movie_input)

        output = Dot(axes=1, name='user_rating_prediction')([user_emb, movie_emb])

        model = Model(
            inputs=[user_input, movie_input],
            outputs=output,
            name='Embedding_Dot_Product_Model'
        )

        model.compile(
            loss=loss,
            optimizer=Adam(lr)
        )

        return model
