import os
import time

from pymorphy3 import MorphAnalyzer
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType

from spark_lp import Lang
from spark_lp.utils import get_stop_words

# Standard speed test from master's thesis, relying on UDF to process the text via pymorphy3
if __name__ == "__main__":
    jars_dir = "./jars_old"  # Change this to your actual directory
    jars = ",".join([os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")])
    spark = (
        SparkSession.builder
            .appName("speedTest-RAPIDS")
            # .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.jars", jars)
            # .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.executorEnv.TZ", "UTC")
            .config("spark.driverEnv.TZ", "UTC")
            .config("spark.rapids.sql.explain", "ALL")
            .getOrCreate()
    )

    spark.conf.set("spark.rapids.sql.enabled",True)

    # df = spark.readStream.schema(
    #     "author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING"
    # ).parquet("data_parquet/")

    # df = df.repartition(2)
    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet/")

    clean_text = lambda c: F.regexp_replace(
        F.regexp_replace(F.col(c), "\n+", " "),  # Replace multiple newlines with a space
        "\\s+", " "  # Replace multiple spaces with a single space
    )

    split_sentences = lambda c: F.split(F.col(c), '\\s*[.!?]+\\s*')
    split_words = lambda c: F.expr(f"transform({c}, x -> split(x, '[,:; \\\\-—\\\"]+'))")

    morph = MorphAnalyzer(lang="uk")
    normalize_flat_fn = lambda words: [morph.parse(word.lower())[0].normal_form for word in words]
    normalize_flat_udf = F.udf(normalize_flat_fn, ArrayType(StringType()))
    normalize_word_udf = F.udf(lambda sents: [normalize_flat_fn(words) for words in sents], ArrayType(ArrayType(StringType())))
    normalize = lambda c: normalize_word_udf(F.col(c))
    normalize_flat = lambda c: normalize_flat_udf(F.col(c))

    stop_words = get_stop_words(Lang('uk'))
    broadcast_stop_words = spark.sparkContext.broadcast(stop_words)
    filter_stop = lambda c: F.array_except(F.col(c), F.array([F.lit(w) for w in broadcast_stop_words.value]))

    df = (
        df
        .withColumn("body", clean_text("body"))
        .withColumn("title", clean_text("title"))
        .withColumn("body_vec", split_sentences("body"))
        .withColumn("title_vec", split_sentences("title"))
        .withColumn("body_vec", split_words("body_vec"))
        .withColumn("title_vec", split_words("title_vec"))
        .withColumn("body_vec", F.flatten("body_vec"))
        .withColumn("title_vec", F.flatten("title_vec"))
        .withColumn("body_vec", filter_stop("body_vec"))
        .withColumn("title_vec", filter_stop("title_vec"))
        .withColumn("body_vec", normalize_flat("body_vec"))
        .withColumn("title_vec", normalize_flat("title_vec"))
    )

    start = time.time()
    df.write.mode("overwrite").parquet("benchmark_output/")
    end = time.time()
    print(f"⏱ Time taken: {end - start:.2f} seconds")
    df.explain()


    # df.writeStream.format("console").start().awaitTermination()
