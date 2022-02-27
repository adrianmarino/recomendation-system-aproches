from tensorflow.keras.layers import Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from recommendation.model.build_fn import embedding_input


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
        user_input, user_emb = embedding_input(n_users, n_factors, 'users', l2_delta=l2_delta)
        movie_input, movie_emb = embedding_input(n_movies, n_factors, 'movies', l2_delta=l2_delta)

        output = Dot(axes=1, name='user_rating_prediction')([user_emb, movie_emb])

        model = Model(
            inputs=[user_input, movie_input],
            outputs=output,
            name='Embedding_Dot_Product_Model'
        )

        model.compile(loss=loss, optimizer=Adam(lr))

        return model
