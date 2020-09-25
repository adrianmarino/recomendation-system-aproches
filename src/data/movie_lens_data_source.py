import glob
import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import tensorflow.keras as keras

from data.dataset import Dataset


class MovieLensDataSource:
    def __init__(self):
        url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zipped_file = keras.utils.get_file(
            "ml-latest-small.zip", url, extract=False
        )
        keras_datasets_path = Path(zipped_file).parents[0]
        self.__dataset_path = keras_datasets_path / "ml-latest-small"

        if not self.__dataset_path.exists():
            with ZipFile(zipped_file, "r") as zip:
                # Extract files
                print("Extracting all the files now...")
                zip.extractall(path=keras_datasets_path)
                print("Done!")

    def files(self):
        path = str(self.__dataset_path / '*.csv')
        return [os.path.basename(f) for f in glob.glob(path)]

    def get_df(self, filename='ratings.csv'):
        return pd.read_csv(self.__dataset_path / filename)

    def dataset(self):
        ratings = self.get_df('ratings.csv')
        movies = self.get_df('movies.csv')
        tags = self.get_df('tags.csv')
        links = self.get_df('links.csv')
        return Dataset(ratings, movies, tags, links)
