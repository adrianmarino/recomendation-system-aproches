import findspark
import pyspark
from pyspark.sql import SparkSession

def read_csv(session, path):
    return session.read.csv(path, inferSchema=True, header=True, mode="DROPMALFORMED").cache()

def write_csv(data_frame, path, sep=','):
    data_frame.write.format('csv').option('header',True).mode('overwrite').option('sep', sep).save(path)

class SparkSessionFactory:
    @staticmethod
    def create():
        findspark.init()

        return SparkSession.builder \
            .master("local") \
            .config("spark.driver.extraClassPath", '/opt/apache-spark/jars/rapids-4-spark_2.12-0.2.0.jar;/opt/apache-spark/jars/cudf-0.15-cuda11.jar') \
            .config('spark.rapids.sql.incompatibleOps.enabled', 'true') \
            .config('spark.rapids.sql.enabled','true') \
            .config('spark.rapids.sql.explain', 'ALL') \
            .config('spark.executor.resource.gpu.amount', '1') \
            .config('spark.rapids.sql.batchSizeBytes', '4G') \
            .config('spark.rapids.sql.reader.batchSizeBytes', '4G') \
            .config("spark.driver.maxResultSize", "16G") \
            .config("spark.executor.memory", "16G") \
            .config('spark.driver.memory', '16G') \
            .getOrCreate()
    
    
