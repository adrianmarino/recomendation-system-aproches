from tensorflow.keras.layers import Input, Dot, Add, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import EmbeddingLayer


class EmbeddingBiasesDotModelFactory:
    @staticmethod
    def create(
            n_users,
            n_movies,
            n_factors,
            min_rating,
            max_rating,
            lr=0.001,
            l2_delta=1e-6,
            loss='mean_squared_error'
    ):
        user_input = Input(shape=(1,), name='users_idx')
        movie_input = Input(shape=(1,), name='movies_idx')

        user_emb = EmbeddingLayer(n_users, n_factors, l2_delta=l2_delta, name='users_embedding')(user_input)
        movie_emb = EmbeddingLayer(n_movies, n_factors, l2_delta=l2_delta, name='movies_embedding')(movie_input)

        dot = Dot(axes=1, name='dot_product')([user_emb, movie_emb])

        user_bias = EmbeddingLayer(n_users, 1, name='users_biases')(user_input)
        movie_bias = EmbeddingLayer(n_movies, 1, name='movie_biases')(movie_input)

        # Sum dot result y both biases
        output = Add(name='Add')([dot, user_bias, movie_bias])

        # Apply sigmode activation function
        output = Activation('sigmoid', name='sigmoid_activation')(output)

        # Normalize out between 0..1
        output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating, name='user_rating_prediction')(output)

        model = Model(
            inputs=[user_input, movie_input],
            outputs=output,
            name='Embedding_Dot_Product_Plus_Biases_Model'
        )

        model.compile(
            loss=loss,
            optimizer=Adam(lr=lr)
        )

        return model
