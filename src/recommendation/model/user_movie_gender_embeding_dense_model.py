from tensorflow.keras.layers import Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from recommendation.model.build_fn import dense_layers, embedding_input


class UserMovieGenderEmbeddingDenseModelFactory:
    @staticmethod
    def create(
            n_users,
            n_movies,
            n_genders,
            min_rating,
            max_rating,
            user_n_min_factors=50,
            movie_n_min_factors=50,
            lr=0.001,
            units=[600, 200],
            dropout=[0, 0],
            loss='mean_squared_error'
    ):
        # Create feature inputs...
        user_input, user_emb = embedding_input(n_users, user_n_min_factors, 'users')
        movie_input, movie_emb = embedding_input(n_movies, movie_n_min_factors, 'movies')
        gender_input = Input(shape=(n_genders,), name='genders')

        net = Concatenate()([user_emb, movie_emb, gender_input])
        if dropout[0] > 0:
            net = Dropout(dropout[0])(net)

        net = dense_layers(net, units, dropout, min_rating, max_rating)

        model = Model(
            inputs=[user_input, movie_input, gender_input],
            outputs=net,
            name='Embedding_Feature_Layers_Plus_Dense_Layer_Model'
        )

        model.compile(loss=loss, optimizer=Adam(lr))

        return model
