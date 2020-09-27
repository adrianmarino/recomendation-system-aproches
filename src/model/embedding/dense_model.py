from tensorflow.keras.layers import Input, Activation, Lambda, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import EmbeddingLayer


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
        user_input = Input(shape=(1,), name='users_idx')
        movie_input = Input(shape=(1,), name='movies_idx')

        user_emb = EmbeddingLayer(n_users, n_factors, name='users_embedding')(user_input)
        movie_emb = EmbeddingLayer(n_movies, n_factors, name='movies_embedding')(movie_input)

        x = Concatenate()([user_emb, movie_emb])
        if dropout[0] > 0:
            x = Dropout(dropout[0])(x)

        for layer_units, layer_dropout in zip(units, dropout):
            x = Dense(layer_units, kernel_initializer='he_normal')(x)
            x = Activation('relu')(x)
            if layer_dropout > 0:
                x = Dropout(layer_dropout)(x)

        x = Dense(1, kernel_initializer='he_normal')(x)
        x = Activation('sigmoid', name='sigmoid_activation')(x)
        x = Lambda(lambda p: p * (max_rating - min_rating) + min_rating, name='user_rating_prediction')(x)

        model = Model(
            inputs=[user_input, movie_input],
            outputs=x,
            name='Embedding_Feature_Layers_Plus_Dense_Layer_Model'
        )

        model.compile(
            loss=loss,
            optimizer=Adam(lr)
        )

        return model
