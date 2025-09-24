import time

from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F

# Modified speed test for CUDA, relying on UDF to process the text via JNI/C++ custom CUDA kernel
if __name__ == "__main__":
    spark = (
        SparkSession
            .builder
            .appName("speedTest")
            .getOrCreate()
    )

    spark.udf.registerJavaFunction("morph_analyzer", "org.example.MorphAnalyzerUDF")
    spark.udf.registerJavaFunction("morfologik", "org.example.MorfologikUDF")
    spark.udf.registerJavaFunction("gpu", "org.example.GPUUDF")

    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet_big/")#.coalesce(16)

    # OLD (OVER 9 MINUTES)
    # df = df.withColumn("sentence_id", F.monotonically_increasing_id())
    #
    # # Explode body
    # df_body = (
    #     df
    #     .select(
    #         col("sentence_id"),
    #         F.explode(F.split(col("body"), "\\s+")).alias("word"),
    #     )
    #     .withColumn("body_vec", F.expr("gpu(word)"))
    #     .groupBy("sentence_id")
    #     .agg(F.concat_ws(" ", F.collect_list("body_vec")).alias("body_vec"))
    # )
    #
    # # Explode title
    # df_title = (
    #     df.select(
    #         col("sentence_id"),
    #         F.explode(F.split(col("title"), "\\s+")).alias("word"),
    #     )
    #     .withColumn("title_vec", F.expr("gpu(word)"))
    #     .groupBy("sentence_id")
    #     .agg(F.concat_ws(" ", F.collect_list("title_vec")).alias("title_vec"))
    # )
    #
    # df = df.join(df_body, on="sentence_id", how="left").join(df_title, on="sentence_id", how="left")

    # NEW (5 MINUTES, BUT STILL SLOW)
    # df = df.withColumn(
    #     "body_vec",
    #     F.expr("concat_ws(' ', transform(split(body, '\\\\s+'), x -> gpu(x)))")
    # ).withColumn(
    #     "title_vec",
    #     F.expr("concat_ws(' ', transform(split(title, '\\\\s+'), x -> gpu(x)))")
    # )

    # NEW (10 SECONDS)
    df = df.withColumn(
        "body_vec",
        F.expr("gpu(body)")
    ).withColumn(
        "title_vec",
        F.expr("gpu(title)")
    )

    start = time.time()
    df.write.mode("overwrite").parquet("benchmark_output/")
    end = time.time()
    print(f"⏱ Time taken: {end - start:.2f} seconds")
    df.explain()
    df.show()

    print("💤 Sleeping to keep Spark UI alive at http://localhost:4040")
    time.sleep(99999)  # or however long you want
