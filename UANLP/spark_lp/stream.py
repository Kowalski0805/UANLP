from time import sleep, time

from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, lit
from pyspark.mllib.feature import HashingTF
from spark_lp.text_ssdf import TextRDD
from spark_lp.text import Text
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from elasticsearch import Elasticsearch, helpers
from collections import deque

from spark_lp.utils import split_to_words

class TextStreaming:
    def __init__(self) -> None:
        pass

    def createElasticClient(self, config):
        es = Elasticsearch([{'host': 'elasticsearch', 'port': '9200'}])

    def stream(self, input_config, output_config):
        spark = SparkSession \
            .builder \
            .appName("speedTest") \
            .master("spark://spark:7077") \
            .config("spark.executor.heartbeatInterval","3500s") \
            .config("spark.network.timeout","3600s") \
            .config("spark.shuffle.registration.timeout",15000) \
            .getOrCreate()

        df = spark.readStream.schema("author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING").format("json").option("path", "data/").load()

        text = TextRDD(spark, df)
        text.process()


        # text.words.writeStream.format("console").start().awaitTermination()
        text.words.writeStream.foreachBatch(self.handleRow).start().awaitTermination()
    
    def handleRow(self, d, i):
        d.persist()
        rows = d.withColumn("ingestion_time", current_timestamp()) \
                .withColumn("_id", col("link")) \
                .withColumn("_op_type", lit("create")) \
                .rdd.map(lambda r: r.asDict(True)).collect()
        deque(helpers.parallel_bulk(es, rows, index="news", ignore_status=409), maxlen=0)
        # res = helpers.bulk()
        print("Batch #"+str(i)+" uploaded")
        d.unpersist()
