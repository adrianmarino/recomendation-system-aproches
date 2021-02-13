import findspark
from pyspark.sql import SparkSession


class SparkSessionFactory:
    @staticmethod
    def create():
        findspark.init()
        return SparkSession.builder \
            .master("local[*]") \
            .appName("recommendations") \
            .config('spark.executor.memory', '1G') \
            .config('spark.driver.memory','4G') \
            .config('spark.executor.instances', 12) \
            .getOrCreate()
    
    @staticmethod
    def connect_to(cluster_host):
        findspark.init()
    
        return SparkSession.builder \
            .master(cluster_host) \
            .appName("recommendations") \
            .getOrCreate()
