import os
from time import sleep, time

from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, lit
from pyspark.mllib.feature import HashingTF
from spark_lp.text_ssdf import TextDataFrame, process_udf
from spark_lp.text import Text
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from elasticsearch import Elasticsearch, helpers
from collections import deque

from spark_lp.utils import split_to_words

# Standard speed test from master's thesis, relying on UDF to process the text via pymorphy3
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("speedTest") \
        .master("local[*]") \
        .config("spark.executor.heartbeatInterval", "3500s") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.shuffle.registration.timeout", 15000) \
        .getOrCreate()

    # es = Elasticsearch(
    #     [{'host': 'es01', 'port': 9200, 'scheme': 'https'}],
    #     basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")),
    #     ca_certs="/opt/certs/ca/ca.crt")

    spark.udf.register("normalize", process_udf)
    df = spark.readStream.schema(
        "author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING"
    ).format("json").option("path", "data/").load()
    # df = spark.readStream.format("kafka") \
    # .option("kafka.bootstrap.servers", "kafka:9092") \
    # .option("subscribe", "topic1") \
    # .load()

    # def handleRow(d, i):
    #     d.persist()
    #     rows = d.withColumn("ingestion_time", current_timestamp()) \
    #         .withColumn("_id", col("link")) \
    #         .withColumn("_op_type", lit("create")) \
    #         .rdd.map(lambda r: r.asDict(True)).collect()
    #     deque(helpers.parallel_bulk(es, rows, index="news", ignore_status=409), maxlen=0)
    #     # res = helpers.bulk()
    #     print("Batch #" + str(i) + " uploaded")
    #     d.unpersist()


    df = df \
        .withColumn("body_vec", F.expr("normalize(body)")) \
        .withColumn("title_vec", F.expr("normalize(title)"))

    df.writeStream.format("console").start().awaitTermination()
    # df.writeStream.foreachBatch(handleRow).start().awaitTermination()
    # text = TextDataFrame(spark, df)
    # text.process_once()
    # text.words.writeStream.format("console").start().awaitTermination()
    # text.words.writeStream.foreachBatch(handleRow).start().awaitTermination()
