import pandas as pd

from model.embedding.simmilarity_model import SimilarityModel
from recommendation.model import Model
from recommendation.recommender.value_index_mapper import ValueIndexMapper


def revert(array): return array[::-1]


class Recommender:

    def __init__(self, model, dataset):
        self.model = Model(model)
        self.dataset = dataset
        self.user_converter = ValueIndexMapper(dataset.ratings(), value_col='userId', index_col='user')
        self.movie_converter = ValueIndexMapper(dataset.ratings(), value_col='movieId', index_col='movie')

    def users_similar_to(self, user_id, limit=10):
        user_idx = self.user_converter.to_index(user_id)

        similarity_model = SimilarityModel(
            self.model.user_embeddings_matrix,
            neighbors_count=limit
        )
        similar_indexes = similarity_model.similar(user_idx)

        return self.user_converter.to_values(similar_indexes)

    def movies_similar_to(self, movie_id, limit=10):
        movie_idx = self.movie_converter.to_index(movie_id)

        similarity_model = SimilarityModel(
            self.model.movie_embeddings_matrix,
            neighbors_count=limit
        )
        similar_indexes = similarity_model.similar(movie_idx)
        similar_ids = self.movie_converter.to_values(similar_indexes)

        return self.dataset.movies_by_ids(similar_ids)

    def top_movies_by_user_id(self, user_id, limit=10):
        user_idx = self.user_converter.to_index(user_id)

        user_ratings = self.model.predict_ratings(user_idx)

        movies = self.dataset.movie_idx_id()

        user_ratings = pd.merge(user_ratings, movies, how='right', on='movie')

        user_ratings = pd.merge(user_ratings, self.dataset.movies(), how='left', on='movieId')
        user_ratings = user_ratings[['predicted_rating', 'title', 'movieId', 'movie']]
        user_ratings = user_ratings.sort_values(by=['predicted_rating'], inplace=False, ascending=False)

        return user_ratings if limit == -1 else user_ratings[:limit]
