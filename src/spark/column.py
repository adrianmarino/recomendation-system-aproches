import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer


class Column:
    @staticmethod
    def copy(session, df):
        schema = df.schema
        pdf = df.toPandas()
        rdf = session.createDataFrame(pdf, schema=schema)
        del pdf
        return rdf

    @classmethod
    def sequence(cls, session, df, input_col, output_col):
        return cls.copy(session, ColumnSequencer(input_col, output_col).perform(df))


class ColumnSequencer:
    def __init__(self, input_col, output_col):
        self.sequencer = StringIndexer(inputCol=input_col, outputCol=output_col)
        self.input_col = input_col
        self.output_col = output_col

    def perform(self, df):
        return self.sequencer \
            .fit(df) \
            .transform(df) \
            .withColumn(self.output_col, f.col(self.output_col).cast("integer"))
