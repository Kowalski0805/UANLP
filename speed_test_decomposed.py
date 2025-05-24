from pyspark.sql.session import SparkSession
from spark_lp.text_ssdf2 import split_sentences, split_words, norm_sent, filter_stop
import pyspark.sql.functions as F

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

    spark.udf.register("split_to_sentences", split_sentences)
    spark.udf.register("split_to_words", split_words)
    spark.udf.register("norm_sent", norm_sent)
    spark.udf.register("filter_stop", filter_stop)
    spark.udf.register("process_body", lambda body: filter_stop(norm_sent(split_words(split_sentences(body)))))

    df = spark.readStream.schema(
        "author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING"
    ).format("json").option("path", "data/").load()

    df = df \
        .withColumn("body_vec", F.expr("process_body(body)")) \
        .withColumn("title_vec", F.expr("process_body(title)"))
    # df = df \
    #     .withColumn("body_vec", F.expr("split_to_sentences(body)")) \
    #     .withColumn("title_vec", F.expr("split_to_sentences(title)")) \
    #     .withColumn("body_vec", F.expr("split_to_words(body_vec)")) \
    #     .withColumn("title_vec", F.expr("split_to_words(title_vec)")) \
    #     .withColumn("body_vec", F.expr("norm_sent(body_vec)")) \
    #     .withColumn("title_vec", F.expr("norm_sent(title_vec)")) \
    #     .withColumn("body_vec", F.expr("filter_stop(body_vec)")) \
    #     .withColumn("title_vec", F.expr("filter_stop(title_vec)"))
    df.writeStream.format("console").start().awaitTermination()
    # df.writeStream.foreachBatch(handleRow).start().awaitTermination()
    # text = TextDataFrame(spark, df)
    # text.process_once()
    # text.words.writeStream.format("console").start().awaitTermination()
    # text.words.writeStream.foreachBatch(handleRow).start().awaitTermination()
