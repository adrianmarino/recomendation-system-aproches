from tensorflow.keras.layers import Input, Dot, Add, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import EmbeddingLayer
from recommendation.model.build_fn import biased_embedding_input


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
        user_input, user_emb, user_bias = biased_embedding_input(
            n_users,
            n_factors,
            'users',
            l2_delta=l2_delta
        )

        movie_input, movie_emb, movie_bias = EmbeddingLayer(
            n_movies,
            n_factors,
            'movies',
            l2_delta=l2_delta
        )

        # Sum dot result y both biases
        dot = Dot(axes=1, name='dot_product')([user_emb, movie_emb])
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

        model.compile(loss=loss, optimizer=Adam(lr=lr))

        return model
