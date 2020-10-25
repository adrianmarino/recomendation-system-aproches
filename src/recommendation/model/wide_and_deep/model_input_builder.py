import tensorflow as tf
from tensorflow import feature_column

from model.feature_column import FeatureColumn

fc = feature_column


class ModelInputBuilder:
    def __init__(self):
        self.__inputs = {}
        self.__feature_columns = []

    def num(self, name):
        column = FeatureColumn.num(name)
        self.__feature_columns.append(column)
        self.input(name, tf.int64)
        return column

    def num_bucketized(self, name, boundaries):
        column = FeatureColumn.num_bucketized(name, boundaries)
        self.__feature_columns.append(column)
        self.input(name, tf.string)
        return column

    def cat_id(self, name, ids_count, unknown_value):
        column = FeatureColumn.cat_id(name, ids_count, unknown_value)
        self.__feature_columns.append(column)
        self.input(name, tf.int64)

        return

    def cat_one_hot(self, name, values):
        cat_col = fc.categorical_column_with_vocabulary_list(name, values)
        self.__feature_columns.append(fc.indicator_column(cat_col))
        self.input(name, tf.string)
        return cat_col

    def cat_one_hot_crossed(self, columns, hash_bucket_size=1000):
        column = FeatureColumn.cat_one_hot_crossed(columns, hash_bucket_size)
        self.__feature_columns.append(column)
        return column

    def cat_embedding(self, name, values, dimension=8):
        column = FeatureColumn.cat_embedding(name, values, dimension)
        self.__feature_columns.append(column)
        self.input(name, tf.string)
        return column

    def input(self, name, type):
        if name not in self.__inputs:
            self.__inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=type)

    def inputs_dic(self): return self.__inputs

    def inputs(self): return self.__inputs.values()

    def feature_columns(self): return self.__feature_columns
