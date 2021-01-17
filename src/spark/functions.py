import numpy as np
import pyspark.sql as s
import pyspark.sql.functions as f


def read_csv(session, path):
    return session.read.csv(path, inferSchema=True, header=True, mode="DROPMALFORMED")


def write_csv(df, path, sep=','):
    df.write.format('csv').option('header', True).mode('overwrite').option('sep', sep).save(path)


def column_values(df, column):
    return df.select(column).rdd.map(lambda x: x[0]).collect()


def shuffle_df(df):
    return df.orderBy(f.rand())


def train_test_split(df, test_size=0.3, seed=None):
    df = shuffle_df(df)
    return df.randomSplit([1 - test_size, test_size], seed)


def add_seq_col(df, column_name='seq'):
    w = s.Window().partitionBy(f.lit('a')).orderBy(f.lit('a'))
    return df.withColumn(column_name, f.row_number().over(w) - 1).sort(column_name)


def get_columns(df, column_names):
    df = df.toPandas()
    return np.array([df[c].values for c in column_names])


def get_rows(df, column_names):
    return np.array(df.select(column_names).rdd.map(tuple).collect())
