from pandas import np

from model.recommender.simmilarity_model import SimilarityModel


def revert(array): return array[::-1]


class Recommender:
    def __init__(self, model):
        self.model = model

        movie_embeddings_layer = model.layers[3]
        self.movie_embeddings_matrix = movie_embeddings_layer.get_weights()[0]

        user_embeddings_layer = model.layers[2]
        self.user_embeddings_matrix = user_embeddings_layer.get_weights()[0]

    def users_similar_to(self, user_idx, count=10):
        similarity_model = SimilarityModel(
            self.user_embeddings_matrix,
            neighbors_count=count
        )
        return similarity_model.similar(user_idx)

    def movies_similar_to(self, movie_idx, count=10):
        similarity_model = SimilarityModel(
            self.movie_embeddings_matrix,
            neighbors_count=count
        )
        return similarity_model.similar(movie_idx)

    def recommended_movies_for(self, user_idx):
        x = self.__to_user_movie_input(user_idx, len(self.movie_embeddings_matrix))

        user_ratings = self.model.predict_on_batch(x)
        user_ratings = user_ratings.reshape((1, -1))[0]

        movie_idxs_orderred_by_ratings_desc = revert(np.argsort(user_ratings))

        return movie_idxs_orderred_by_ratings_desc

    def __to_user_movie_input(self, user_idx, movies_count):
        movie_idxs = np.linspace(0, movies_count - 1, movies_count, dtype=int)
        user_idxs = np.repeat(user_idx, movies_count)
        return [user_idxs, movie_idxs]
