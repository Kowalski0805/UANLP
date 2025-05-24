import os
import time

from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, lit
from pyspark.mllib.feature import HashingTF
from spark_lp.text_ssdf import TextDataFrame
from spark_lp.text import Text
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from elasticsearch import Elasticsearch, helpers
from collections import deque

from spark_lp.utils import split_to_words

# Modified speed test for JVM, relying on UDF to process the text via Java (Jmorphy or Morfologik)
if __name__ == "__main__":
    jars_dir = "./jars"  # Change this to your actual directory
    jars = ",".join([os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")])
    native_path = "./native"
    spark = (
        SparkSession
        .builder
        .appName("speedTest")
        # .master("spark://spark:7077") \
        .config("spark.driver.memory", "4g")
        .config("spark.shuffle.registration.timeout", 15000)
        .config("spark.jars", jars)
        .config("spark.executor.extraLibraryPath", native_path)
        .config("spark.driver.extraLibraryPath", native_path)
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.concurrentGpuTasks", "1")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.executorEnv.TZ", "UTC")
        .config("spark.driverEnv.TZ", "UTC")
        .config("spark.rapids.sql.explain", "ALL")
        .getOrCreate()
        # .config("spark.rapids.sql.udfCompiler.enabled", "true") \
        # .config("spark.executor.resource.gpu.amount", "1") \
        # .config("spark.executor.resource.gpu.discoveryScript", "/spark_lp/getGpusResources.sh") \
        # .config("spark.task.resource.gpu.amount", "1") \
        # .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", "0") \
        # .config("spark.driver.resource.gpu.amount", "1") \
        # .config("spark.driver.resource.gpu.discoveryScript", "/spark_lp/getGpusResources.sh") \
    )

    spark.udf.registerJavaFunction("morph_analyzer", "org.example.MorphAnalyzerUDF")
    spark.udf.registerJavaFunction("morfologik", "org.example.MorfologikUDF")
    spark.udf.registerJavaFunction("gpu", "org.example.GPUUDF")
    # es = Elasticsearch(
    #     [{'host': 'es01', 'port': 9200, 'scheme': 'https'}],
    #     basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")),
    #     ca_certs="/opt/certs/ca/ca.crt")

    # df = spark.readStream.schema(
    #     "author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING"
    # ).format("json").option("path", "data/").load()
    # df = spark.readStream.schema(
    #     "author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING"
    # ).parquet("data_parquet/")
    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet/")
    # df = spark.readStream.format("kafka") \
    # .option("kafka.bootstrap.servers", "kafka:9092") \
    # .option("subscribe", "topic1") \
    # .load()

    # df = df \
    #     .withColumn("body_vec", F.expr("morfologik(body)")) \
    #     .withColumn("title_vec", F.expr("morfologik(title)"))
    df = df \
        .withColumn("body_vec", F.expr("gpu(body)")) \
        .withColumn("title_vec", F.expr("gpu(title)"))

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


    # df.writeStream.format("console").start().awaitTermination()
    start = time.time()
    df.write.mode("overwrite").parquet("benchmark_output/")
    end = time.time()
    print(f"⏱ Time taken: {end - start:.2f} seconds")
    df.explain()
    df.show()
    # df.writeStream.foreachBatch(handleRow).start().awaitTermination()
