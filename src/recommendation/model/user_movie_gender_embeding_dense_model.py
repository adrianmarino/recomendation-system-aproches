from tensorflow.keras.layers import Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import create_embedding_input
from recommendation.model.build_fn import dense_layers


class UserMovieGenderEmbeddingDenseModelFactory:
    @staticmethod
    def create(
            self,
            n_users,
            n_movies,
            genders,
            min_rating,
            max_rating,
            user_n_factors=50,
            movie_n_factors=50,
            lr=0.001,
            units=[600, 200],
            dropout=[0, 0],
            loss='mean_squared_error'
    ):
        # Create feature inputs...
        user_emb = create_embedding_input(n_users, user_n_factors, 'users')
        movie_emb = create_embedding_input(n_movies, movie_n_factors, 'movies')
        gender_inputs = [Input(shape=(1,), name=g) for g in genders]

        inputs = [user_emb, movie_emb] + gender_inputs
        net = Concatenate()(inputs)
        if dropout[0] > 0:
            net = Dropout(dropout[0])(net)

        net = dense_layers(net, units, dropout, min_rating, max_rating)

        model = Model(
            inputs=inputs,
            outputs=net,
            name='Embedding_Feature_Layers_Plus_Dense_Layer_Model'
        )

        model.compile(loss=loss, optimizer=Adam(lr))

        return model
