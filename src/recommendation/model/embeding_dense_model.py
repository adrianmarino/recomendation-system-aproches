from tensorflow.keras.layers import Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from recommendation.model.build_fn import embedding_input, ratting_dense_layers


class EmbeddingDenseModelFactory:
    @staticmethod
    def create(
            n_users,
            n_movies,
            n_factors,
            min_rating,
            max_rating,
            lr=0.001,
            units=[300, 100],
            dropout=[0, 0],
            loss='mean_squared_error'
    ):
        user_input, user_emb = embedding_input(n_users, n_factors, 'users')
        movie_input, movie_emb = embedding_input(n_movies, n_factors, 'movies')

        net = Concatenate()([user_emb, movie_emb])
        if dropout[0] > 0:
            net = Dropout(dropout[0])(net)

        net = ratting_dense_layers(net, units, dropout, min_rating, max_rating)

        model = Model(
            inputs=[user_input, movie_input],
            outputs=net,
            name='Embedding_Feature_Layers_Plus_Dense_Layer_Model'
        )

        model.compile(loss=loss, optimizer=Adam(lr))

        return model
