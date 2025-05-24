import os
import time

from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F

# Standard speed test from master's thesis, relying on UDF to process the text via pymorphy3
if __name__ == "__main__":
    jars_dir = "./jars_old"  # Change this to your actual directory
    jars = ",".join([os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")])
    spark = (
        SparkSession.builder
            .appName("speedTest-RAPIDS")
            # .master("local[*]")
            .config("spark.driver.memory", "8g")
            .config("spark.jars", jars)
            # .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.executorEnv.TZ", "UTC")
            .config("spark.driverEnv.TZ", "UTC")
            .config("spark.eventLog.gcMetrics.youngGenerationGarbageCollectors", "G1 Young Generation")
            .config("spark.eventLog.gcMetrics.oldGenerationGarbageCollectors", "G1 Old Generation")
            # .config("spark.rapids.sql.explain", "ALL")
            .getOrCreate()
    )

    # partitions = 16
    spark.conf.set("spark.rapids.sql.enabled",True)

    morph_df = spark.read.parquet("ukr_morph_dict.parquet")

    # df = spark.readStream.schema(
    #     "author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING"
    # ).parquet("data_parquet/")

    # df = df.repartition(2)
    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet/")
    # df = df.dropDuplicates(["link"])

    clean_text = lambda c: F.regexp_replace(
        F.regexp_replace(F.col(c), "\n+", " "),  # Replace multiple newlines with a space
        "\\s+", " "  # Replace multiple spaces with a single space
    )

    split_sentences = lambda c: F.split(F.col(c), '\\s*[.!?]+\\s*')
    split_words = lambda c: F.expr(f"transform({c}, x -> split(x, '[,:; \\\\-—\\\"]+'))")


    # GPU-supported tokenization
    tokens_df = (df
        .withColumn("body", F.lower("body"))
        .withColumn("tokens", clean_text("body"))
        .withColumn("tokens", split_sentences("tokens"))
        .withColumn("tokens", split_words("tokens"))
    )

    # Explode for word-level join
    tokens_df = tokens_df.select("link", "tokens").withColumn("tokens", F.flatten("tokens")).withColumn("tokens", F.explode("tokens"))
    # tokens_df = tokens_df.repartition(partitions)

    joined_df = tokens_df.join(
        morph_df,
        tokens_df["tokens"] == morph_df["wordform"],
        how="left"
    )

    joined_df = joined_df.withColumn(
        "normalized_token",
        F.coalesce("lemma", "tokens")  # use lemma if found, else original token
    )


    normalized_df = joined_df.repartition("link").groupBy("link").agg(
        F.collect_list("normalized_token").alias("normalized_body")
    )


    df = df.repartition("link").join(normalized_df.repartition("link"), on="link", how="left")

    start = time.time()
    df.write.mode("overwrite").parquet("benchmark_output/")
    # df.select("normalized_body").agg(F.count("*")).show()
    end = time.time()
    print(f"⏱ Time taken: {end - start:.2f} seconds")
    df.explain()
    df.show()


    # df.writeStream.format("console").start().awaitTermination()
