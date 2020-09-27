import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, ratings, movies, tags, links):
        user_enc = LabelEncoder()
        ratings['user'] = user_enc.fit_transform(ratings['userId'].values)

        item_enc = LabelEncoder()
        ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)

        ratings['rating'] = ratings['rating'].values.astype(np.float32)

        self.__ratings = ratings
        self.__movies = movies
        self.__tags = tags
        self.__links = links

    def n_users(self):
        return self.__ratings['user'].nunique()

    def n_movies(self):
        return self.__ratings['movie'].nunique()

    def user_ids(self):
        return self.__ratings['userId'].unique()

    def movie_ids(self):
        return self.__ratings['movieId'].unique()

    def highest_user_ratings(self, limit):
        g = self.__ratings.groupby('userId')['rating'].count()
        top_users = g.sort_values(ascending=False)[:limit]
        return top_users.to_frame()

    def highest_movie_ratings(self, limit):
        g = self.__ratings.groupby('movieId')['rating'].count()
        top_movies = g.sort_values(ascending=False)[:limit]
        return top_movies.to_frame()

    def top_user_vs_movies(self, limit):
        top_users = self.highest_user_ratings(limit)
        top_movies = self.highest_movie_ratings(limit)

        top_r = self.__ratings.join(top_users, rsuffix='_top_user', how='inner', on='userId')
        top_r = top_r.join(top_movies, rsuffix='_top_movie', how='inner', on='movieId')
        top_r = top_r[['userId', 'movieId', 'rating']]

        return pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)

    def min_rating(self): return min(self.__ratings['rating'])

    def max_rating(self): return max(self.__ratings['rating'])

    def movie_idx_id(self):
        movies = self.ratings()[['movie', 'movieId']]
        return movies.drop_duplicates()

    def movies_by_ids(self, ids):
        d =  self.__movies[self.__movies['movieId'].isin(ids)]
        d = pd.merge(d, self.ratings(), how='left', on='movieId')
        return d[['movieId', 'movie', 'title', 'genres']].drop_duplicates()

    def movies(self): return self.__movies

    def ratings(self): return self.__ratings

    def tags(self): return self.__tags

    def links(self): return self.__links

    def top_movies_by_user_id(self, user_id, limit=10):
        user_ratings = self.ratings()[self.ratings()['userId'] == user_id]
        user_ratings = pd.merge(user_ratings, self.movies(), how='left', on='movieId')
        user_ratings = user_ratings[['rating', 'title', 'movieId']]
        user_ratings = user_ratings.sort_values(by=['rating'], inplace=False, ascending=False)
        return user_ratings[:limit]

    def movies_by_title(self, title):
        return self.__movies[self.__movies['title'].str.contains(title)]

    def movie_by_id(self, movie_id):
        return self.__movies[self.__movies['movieId'] == movie_id]

    def rating_of(self, user_idx, movie_idx):
        r = self.__ratings[self.__ratings['user'] == user_idx]
        r = r[r['movie'] == movie_idx]
        r = pd.merge(r, self.movies(), how='left', on='movieId')
        return r[['user', 'movie', 'userId', 'movieId', 'rating', 'title', 'genres']]

    def movie_by_idx(self, movie_idx):
        r = self.__ratings[self.__ratings['movie'] == movie_idx]
        r = pd.merge(r, self.movies(), how='left', on='movieId')
        return r[['userId', 'movieId', 'rating', 'title', 'genres']]
