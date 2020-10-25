import numpy as np
from tensorflow.keras.utils import Sequence

from spark.page_set import PageSet


class DataFrameDataGenerator(Sequence):
    def __init__(
            self,
            data_frame,
            feature_columns,
            label_columns,
            batch_size,
            shuffle=True,
            transform=lambda x, y: (x, y)
    ):
        self.__page_set = PageSet(data_frame, page_size=batch_size, shuffle=shuffle)
        self.__feature_columns = feature_columns
        self.__label_columns = label_columns
        self.__transform = transform

    def __len__(self): return self.__page_set.size()

    def __getitem__(self, index):
        page = self.__page_set[index]

        X = self.__feature_column_values(page)
        y = self.__label_column_values(page)

        return self.__transform(X, y)

    def __feature_column_values(self, df):
        df = df.toPandas()
        return np.array([df[c].values for c in self.__feature_columns])

    def __label_column_values(self, df):
        return np.array(df.select(self.__label_columns).rdd.map(tuple).collect())
