from pyspark.sql.session import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("speedTest") \
        .master("spark://spark:7077") \
        .config("spark.executor.heartbeatInterval", "3500s") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.shuffle.registration.timeout", 15000) \
        .getOrCreate()

    df = spark.read.json("data/", schema="author STRING, body STRING, category STRING, date TIMESTAMP, link STRING, title STRING")
    df.write.json("data_jsonl/")
    # print(df.count())
    # df.show()
