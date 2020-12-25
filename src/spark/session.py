import findspark
from pyspark.sql import SparkSession


class SparkSessionFactory:
    @staticmethod
    def create():
        findspark.init()

        return SparkSession.builder \
            .master("local[*]") \
            .config('spark.executor.resource.gpu.amount', '1') \
            .config('spark.driver.memory', '14G') \
            .getOrCreate()
            #.config("spark.driver.extraClassPath",
            #        '/opt/apache-spark/jars/rapids-4-spark_2.12-0.2.0.jar;/opt/apache-spark/jars/cudf-0.15-cuda11.jar') \
            # .config('spark.rapids.sql.incompatibleOps.enabled', 'true') \
            # .config('spark.rapids.sql.enabled', 'true') \
            # .config('spark.rapids.sql.explain', 'ALL') \

