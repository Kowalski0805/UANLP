from time import time

from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.mllib.feature import HashingTF
from spark_lp.text_rdd import TextRDD
from spark_lp.text import Text
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from elasticsearch import Elasticsearch, helpers
from collections import deque

from spark_lp.utils import split_to_words

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("speedTest") \
        .master("local[6]") \
        .config("spark.executor.heartbeatInterval","3500s") \
        .config("spark.network.timeout","3600s") \
        .config("spark.shuffle.registration.timeout",15000) \
        .getOrCreate()

    es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

    df = spark.readStream.schema("author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING").format("json").option("path", "data/").load()

    text = TextRDD(spark, df)
    text.split_to_sentences()
    text.tokenize()
    text.filter_stop_words()

    def handleRow(d, i):
        d.persist()
        rows = d.rdd.map(lambda r: r.asDict(True)).collect()
        deque(helpers.parallel_bulk(es, rows, index="news"), maxlen=0)
        # res = helpers.bulk()
        print(rows)
        print("Batch #"+str(i)+" uploaded")
        d.unpersist()

    # text.words.writeStream.format("console").start().awaitTermination()
    text.words.writeStream.foreachBatch(handleRow).start().awaitTermination()
