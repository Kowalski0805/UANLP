import os
import time

from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F

# Modified speed test for CUDA, relying on UDF to process the text via JNI/C++ custom CUDA kernel
if __name__ == "__main__":
    spark = (
        SparkSession
        .builder
        .appName("speedTest")
        # .config("spark.rapids.sql.explain", "ALL")
        # .config("spark.rapids.sql.enabled.ops", "")  # Optional but explicit
        # .config("spark.rapids.sql.exec.Enabled", "false")
        # .config("spark.rapids.sql.expression.Enabled", "false")
        # .config("spark.rapids.sql.exec.GenerateExec", "false")       # explode
        # .config("spark.rapids.sql.expression.StringSplit", "false")  # split
        # .config("spark.rapids.sql.expression.ConcatWs", "false")     # concat_ws
        # .config("spark.rapids.sql.expression.CollectList", "false")  # collect_list
        # .config("spark.rapids.sql.expression.SortOrder", "false")
        # .config("spark.rapids.sql.exec.ProjectExec", "true")
        # .config("spark.rapids.sql.exec.AggregateExec", "false")
        # .config("spark.rapids.sql.exec.FileSourceScanExec", "false")
        # .config("spark.rapids.sql.exec.DataWritingCommandExec", "false")
        # .config("spark.rapids.sql.exec.SortMergeJoinExec", "false")
        # .config("spark.rapids.sql.exec.CoalesceExec", "false")
        # .config("spark.rapids.sql.exec.SortExec", "false")
        # .config("spark.rapids.sql.exec.LocalLimitExec", "false")
        # # Enable GPU only for ScalaUDFs
        # .config("spark.rapids.sql.udfCompiler.enabled", "true")
        # .config("spark.rapids.sql.udf.enabled", "true")
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
    df = spark.read.parquet("data_parquet/").coalesce(1)
    # df = spark.readStream.format("kafka") \
    # .option("kafka.bootstrap.servers", "kafka:9092") \
    # .option("subscribe", "topic1") \
    # .load()

    # df = df \
    #     .withColumn("body_vec", F.expr("morfologik(body)")) \
    #     .withColumn("title_vec", F.expr("morfologik(title)"))

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

    import time

    print("💤 Sleeping to keep Spark UI alive at http://localhost:4040")
    time.sleep(99999)  # or however long you want

    # df.writeStream.foreachBatch(handleRow).start().awaitTermination()
