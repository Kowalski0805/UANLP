from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as f

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("batchReadTest") \
        .master("spark://spark:7077") \
        .config("spark.executor.heartbeatInterval", "3500s") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.shuffle.registration.timeout", 15000) \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    spark.conf.set("spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName", "true")

    # Read data from folder
    df = spark.read.schema("author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING") \
        .json("data/")

    # Count records
    record_count = df.count()
    print(f"Total records: {record_count}")

    distinct_title_count = df.select("title").distinct().count()
    print(f"Distinct titles: {distinct_title_count}")

    # Count total words in 'body' column
    word_df = df.withColumn('wordCount', f.size(f.split(f.col('body'), ' '))).select(f.sum('wordCount')).collect()
    print(f"Total words in 'body' column: {word_df[0][0]}")

    # Optionally show some records
    df.show(5)

    spark.stop()
