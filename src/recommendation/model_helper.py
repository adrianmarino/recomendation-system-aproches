import numpy as np
import pandas as pd


class ModelHelper:
    def __init__(self, model):
        self.__model = model

        movie_embeddings_layer = model.layers[3]
        self.movie_embeddings_matrix = movie_embeddings_layer.get_weights()[0]

        user_embeddings_layer = model.layers[2]
        self.user_embeddings_matrix = user_embeddings_layer.get_weights()[0]

    def predict_ratings(self, user_idx):
        input = self.__to_user_movie_input(user_idx, len(self.movie_embeddings_matrix))
        output = self.__model.predict(input)

        return self.__to_user_ratings_output(output)

    def predict_rating(self, user_idx, movie_idx):
        movie_ratings = self.predict_ratings(user_idx=user_idx)
        return movie_ratings[movie_ratings['movie'] == movie_idx]

    def __to_user_ratings_output(self, user_ratings):
        user_ratings = user_ratings.reshape((1, -1))[0]
        user_ratings = pd.DataFrame(data=user_ratings, columns=['predicted_rating'])
        user_ratings.reset_index(inplace=True)
        user_ratings = user_ratings.rename(columns={'index': 'movie'})
        return user_ratings

    def __to_user_movie_input(self, user_idx, movies_count):
        movie_idxs = np.linspace(0, movies_count - 1, movies_count, dtype=int)
        user_idxs = np.repeat(user_idx, movies_count)
        return [user_idxs, movie_idxs]
