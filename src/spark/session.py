import findspark
from pyspark.sql import SparkSession


class SparkSessionFactory:
    @staticmethod
    def create():
        findspark.init()

        session = SparkSession.builder \
            .master("local[*]") \
            .appName("recommendations") \
            .config('spark.executor.resource.gpu.amount', '1') \
            .config('spark.driver.memory', '6G') \
            .config('spark.executor.memory', '4G') \
            .config("spark.driver.extraClassPath", '/opt/apache-spark/jars/;/opt/apachrapids-4-spark_2.12-0.2.0.jar;e-spark/jars/cudf-0.17.jar') \
            .config('spark.rapids.sql.incompatibleOps.enabled', 'true') \
            .config('spark.rapids.sql.enabled', 'true') \
            .config('spark.rapids.sql.explain', 'ALL') \
            .config('spark.executor.instances', 12) \
            .getOrCreate()
        
        session.sparkContext.setLogLevel("ERROR")
    
        return session
    
    @staticmethod
    def connect_to(cluster_host):
        findspark.init()
    
        return SparkSession.builder \
            .master(cluster_host) \
            .appName("recommendations") \
            .getOrCreate()
