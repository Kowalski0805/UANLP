from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.sql import functions as F
import time

def main():
    spark = spark_init()
    create_iceberg_tables(spark)

    with open('output.txt', 'a+') as f:

        start_time = time.time()
        world_cities(spark)
        print(f"[python]: world_cities ran in {time.time() - start_time:.2f} seconds.", file=f)

        start_time = time.time()
        weather_events(spark)
        print(f"[python]: weather_events ran in {time.time() - start_time:.2f} seconds.", file=f)

        start_time = time.time()
        weather_hourly_events(spark)
        print(f"[python]: weather_hourly_events ran in {time.time() - start_time:.2f} seconds.", file=f)

def spark_init():

    spark = SparkSession.builder \
        .appName("IcebergTableCreation") \
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.local.type", "hadoop") \
        .config("spark.sql.catalog.local.warehouse", "C:\spark_training\dwh") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.executor.instances", "6") \
        .config("spark.executor.cores", "8") \
        .config("spark.executor.memory", "6g") \
        .config("spark.executor.memoryOverhead", "3g") \
        .config("spark.driver.memory", "6g") \
        .config("spark.driver.cores", "3") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.optimizer.dynamicPartitionPruning", "enabled") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "8g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()
    return spark

def create_iceberg_tables(spark):
    print("Start creating Iceberg tables")
    programming_lan = ['python', 'scala', 'java']

    for lan in programming_lan:
        spark.sql(f"""
        CREATE DATABASE IF NOT EXISTS worldcities_{lan};
        """)

        spark.sql(f"""
        DROP TABLE IF EXISTS local.worldcities_{lan}.weather_events
        """)

        spark.sql(f"""
        DROP TABLE IF EXISTS local.worldcities_{lan}.world_cities
        """)

        spark.sql(f"""
        DROP TABLE IF EXISTS local.worldcities_{lan}.weather_hourly_events
        """)

        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS local.worldcities_{lan}.weather_events (
            lat DOUBLE,
            lng DOUBLE,
            temperature_ts TIMESTAMP,
            temperature DOUBLE
        ) USING iceberg
        PARTITIONED BY (temperature_dt DATE)
        """)

        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS local.worldcities_{lan}.world_cities (
            country STRING,
            province STRING,
            city STRING,
            capital STRING,
            population BIGINT,
            lat DOUBLE,
            lng DOUBLE
        ) USING iceberg
        PARTITIONED BY (country_iso2 STRING)
        """)

        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS local.worldcities_{lan}.weather_hourly_events (
            lat DOUBLE,
            lng DOUBLE,
            temperature_ts TIMESTAMP,
            temperature DOUBLE,
            population BIGINT,
            country STRING,
            province STRING,
            city STRING
        ) USING iceberg
        PARTITIONED BY (temperature_dt DATE, country_iso2 STRING)
        """)

        print(f"Iceberg tables for {lan} have been created!")

def world_cities(spark):

    tbl_name = "local.worldcities_python.world_cities"
    csv_path = "../input_files/worldcities.csv"

    print(f"Start loading data into {tbl_name} table")

    csv_df = spark.read.csv(csv_path, header=True, inferSchema=True)

    csv_df = csv_df.select(
        col("country"),
        col("admin_name").alias("province"),
        col("city"),
        col("capital"),
        col("population"),
        col("lat"),
        col("lng"),
        col("iso2").alias("country_iso2")
    )

    csv_df.write \
        .format("iceberg") \
        .mode("overwrite") \
        .option("overwrite-mode", "dynamic") \
        .partitionBy("country_iso2") \
        .save(tbl_name)

    print(f"Data have been written to the Iceberg table {tbl_name} successfully!")

def weather_events(spark):

    tbl_name = "local.worldcities_python.weather_events"
    csv_path = "../input_files/weather_data_unified.csv"

    print(f"Start loading data into {tbl_name} table")

    csv_df = spark.read.csv(csv_path, header=True, inferSchema=True)

    csv_df = csv_df.select(
        col("lat"),
        col("lng"),
        col("temperature_2m").alias("temperature"),
        col("date").alias("temperature_ts"),
        to_date(col("date")).alias("temperature_dt")
    )

    csv_df.write \
        .format("iceberg") \
        .mode("overwrite") \
        .option("overwrite-mode", "dynamic") \
        .partitionBy("temperature_dt") \
        .save(tbl_name)

    print(f"Data have been written to the Iceberg table {tbl_name} successfully!")

def weather_hourly_events(spark):

    tbl_name = "local.worldcities_python.weather_hourly_events"
    csv_weather_data = "../input_files/weather_data_unified.csv"
    csv_worldcities_data = "../input_files/worldcities.csv"

    print(f"Start loading data into {tbl_name} table")

    csv_weather_df = spark.read.csv(csv_weather_data, header=True, inferSchema=True)

    csv_worldcities_df = spark.read.csv(csv_worldcities_data, header=True, inferSchema=True)

    csv_weather_df = csv_weather_df.select(
        col("lat"),
        col("lng"),
        col("temperature_2m").alias("temperature"),
        col("date").alias("temperature_ts"),
        to_date(col("date")).alias("temperature_dt")
    )

    csv_weather_df = csv_weather_df.dropDuplicates(["lat",
                                                    "lng",
                                                    "temperature_ts"])

    csv_worldcities_df = csv_worldcities_df.select(
        col("country"),
        col("admin_name").alias("province"),
        col("city"),
        col("population"),
        col("lat"),
        col("lng"),
        col("iso2").alias("country_iso2")
    )
    csv_worldcities_df = F.broadcast(csv_worldcities_df)
    csv_worldcities_df = csv_worldcities_df.dropDuplicates(["lat", "lng"])


    csv_df = csv_weather_df.join(
        csv_worldcities_df,
        (csv_weather_df['lat'] == csv_worldcities_df['lat']) &
        (csv_weather_df['lng'] == csv_worldcities_df['lng']),
        how='inner'
    ).select(csv_weather_df["lat"],
             csv_weather_df["lng"],
             csv_weather_df["temperature_ts"],
             csv_weather_df["temperature"],
             csv_worldcities_df['population'],
             csv_worldcities_df['country'],
             csv_worldcities_df['province'],
             csv_worldcities_df['city'],
             csv_weather_df["temperature_dt"],
             csv_worldcities_df['country_iso2']
             )

    csv_df.write \
        .format("iceberg") \
        .mode("overwrite") \
        .option("overwrite-mode", "dynamic") \
        .partitionBy("temperature_dt", "country_iso2") \
        .save(tbl_name)

    print(f"Data have been written to the Iceberg table {tbl_name} successfully!")

if __name__ == "__main__":
    main()
