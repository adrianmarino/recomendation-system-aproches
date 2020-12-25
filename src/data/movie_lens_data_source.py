import glob
import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import tensorflow.keras as keras

from data.dataset import Dataset
from spark import read_csv

PANDAS_DF = 'pandas_data_frame'

SPARK_DF = 'spark_data_frame'


class MovieLensDataSource:

    @staticmethod
    def sizes():
        return ['ml-latest-small', 'ml-25m', 'ml-latest']

    def __init__(self, size='ml-latest-small'):
        url = f'http://files.grouplens.org/datasets/movielens/{size}.zip'

        zipped_file = keras.utils.get_file(
            f'{size}.zip', url, extract=False
        )
        keras_datasets_path = Path(zipped_file).parents[0]
        self.__dataset_path = keras_datasets_path / size

        if not self.__dataset_path.exists():
            with ZipFile(zipped_file, "r") as zip:
                # Extract files
                print("Extracting all the files now...")
                zip.extractall(path=keras_datasets_path)
                print("Done!")

    def files(self):
        path = str(self.__dataset_path / '*.csv')
        return [os.path.basename(f) for f in glob.glob(path)]

    def file_paths(self):
        path = str(self.__dataset_path / '*.csv')
        return [f for f in glob.glob(path)]

    def get_df(self, filename='ratings.csv', session=None, data_frame_type=PANDAS_DF):
        path = f'{self.__dataset_path}/{filename}'

        if data_frame_type == PANDAS_DF:
            return pd.read_csv(path)
        elif data_frame_type == SPARK_DF:
            return read_csv(session, path)

    def dataset(self):
        ratings = self.get_df('ratings.csv')
        movies = self.get_df('movies.csv')
        tags = self.get_df('tags.csv')
        links = self.get_df('links.csv')
        return Dataset(ratings, movies, tags, links)
