import logging

from tensorflow.keras.utils import Sequence

from spark import get_rows
from spark.page_set import PageSet


class DataFrameDataGenerator(Sequence):
    def __init__(
            self,
            data_frame,
            feature_columns,
            label_columns,
            batch_size,
            shuffle=True,
            name='data-frame'
    ):
        self.__page_set = PageSet(data_frame, page_size=batch_size, shuffle=shuffle)
        self.__feature_columns = feature_columns
        self.__label_columns = label_columns
        self._logger = logging.getLogger(f'{name}-data-generator')

    def __len__(self): return self.__page_set.size()

    def __getitem__(self, index):
        self._logger.debug('index(0-%s): %s', self.__page_set.size()-1, index)
        page = self.__page_set[index]

        X = self._features(page, self.__feature_columns)
        y = self._labels(page, self.__label_columns)

        return (X, y)

    def _labels(self, page, columns):
        return get_rows(page, columns)

    def _features(self, page, columns):
        return get_rows(page, columns)