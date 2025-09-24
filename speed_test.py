import time

from pyspark.sql.functions import udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType

from spark_lp.text_ssdf import process_udf
import pyspark.sql.functions as F

# Standard speed test from master's thesis, relying on UDF to process the text via pymorphy3
if __name__ == "__main__":
    spark = (
        SparkSession
            .builder
            .appName("speedTest")
            # .config("spark.rapids.sql.enabled", "false")
            .getOrCreate()
    )

    @udf(returnType=StringType())
    def noop(text):
        return text

    # spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)
    spark.udf.register("normalize", process_udf)
    spark.udf.register("noop", noop)

    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet_big/")

    # df = df \
    #     .withColumn("body_vec", F.expr("normalize(body)")) \
    #     .withColumn("title_vec", F.expr("normalize(title)"))

    # df = df \
    #     .withColumn("body_vec", F.expr("noop(body)")) \
    #     .withColumn("title_vec", F.expr("noop(title)"))

    df = df \
        .withColumn("body_vec", F.col("body")) \
        .withColumn("title_vec", F.col("title"))

    start = time.time()
    df.write.mode("overwrite").parquet("benchmark_output/")
    end = time.time()
    print(f"⏱ Time taken: {end - start:.2f} seconds")
    df.explain()
    df.show()

    print("💤 Sleeping to keep Spark UI alive at http://localhost:4040")
    time.sleep(99999)  # or however long you want
