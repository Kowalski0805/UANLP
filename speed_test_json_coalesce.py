import os
from time import sleep, time

from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, lit
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from elasticsearch import Elasticsearch, helpers
from collections import deque

from spark_lp.utils import split_to_words

if __name__ == "__main__":
    jars_dir = "./jars"  # Change this to your actual directory
    jars = ",".join([os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")])
    spark = SparkSession \
        .builder \
        .appName("speedTest") \
        .master("spark://spark:7077") \
        .config("spark.executor.heartbeatInterval", "3500s") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.shuffle.registration.timeout", 15000) \
        .config("spark.jars", jars) \
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
        .config("spark.rapids.sql.explain", "ALL") \
        .config("spark.rapids.sql.enabled", "true") \
        .config("spark.rapids.sql.udfCompiler.enabled", "true") \
        .config("spark.executor.resource.gpu.amount", "1") \
        .config("spark.executor.resource.gpu.discoveryScript", "/spark_lp/getGpusResources.sh") \
        .config("spark.task.resource.gpu.amount", "1") \
        .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", "0") \
        .config("spark.driver.resource.gpu.amount", "1") \
        .config("spark.driver.resource.gpu.discoveryScript", "/spark_lp/getGpusResources.sh") \
        .getOrCreate()

    spark.udf.registerJavaFunction("morph_analyzer", "org.example.MorphAnalyzerUDF")
    # spark.udf.registerJavaFunction("morfologik", "org.example.MorfologikUDF")

    es = Elasticsearch(
        [{'host': 'es01', 'port': 9200, 'scheme': 'https'}],
        basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")),
        ca_certs="/opt/certs/ca/ca.crt")

    df = spark.read.parquet("data_parquet/", schema="author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING")
    # df = spark.readStream.format("kafka") \
    # .option("kafka.bootstrap.servers", "kafka:9092") \
    # .option("subscribe", "topic1") \
    # .load()

    # df = df \
    #     .withColumn("body_vec", F.expr("morfologik(body)")) \
    #     .withColumn("title_vec", F.expr("morfologik(title)"))
    df = df.withColumn("body_words", F.expr("split(body, ' ')")) \
        .withColumn("body_word", F.explode(F.col("body_words")))  # 🚀 Each word becomes its own row!
    # df = df.withColumn("body_processed_word", F.expr("morph_analyzer(body_word)"))  # ✅ RAPIDS handles STRING
    # df = df.groupBy("author", "category", "date", "link", "title") \
    #     .agg(F.array_join(F.collect_list("body_processed_word"), " ").alias("processed_body"))

    # df = df \
    #     .withColumn("body_vec", F.expr("morph_analyzer(body)")) \
    #     .withColumn("title_vec", F.expr("morph_analyzer(title)"))

    def handleRow(d, i):
        d.persist()
        rows = d.withColumn("ingestion_time", current_timestamp()) \
            .withColumn("_id", col("link")) \
            .withColumn("_op_type", lit("create")) \
            .rdd.map(lambda r: r.asDict(True)).collect()
        deque(helpers.parallel_bulk(es, rows, index="news", ignore_status=409), maxlen=0)
        # res = helpers.bulk()
        print("Batch #" + str(i) + " uploaded")
        d.unpersist()


    # df.writeStream.format("console").start().awaitTermination()
    # df.writeStream.foreachBatch(handleRow).start().awaitTermination()
    print(df.count())
    df.show()
