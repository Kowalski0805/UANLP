from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F

if __name__ == "__main__":
    spark = (
        SparkSession
            .builder
            .appName("parquetToTxt")
            .getOrCreate()
    )

    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet_big/")

    (
        df
        .select(F.col("body"))
        .filter(F.col("body").isNotNull())
        .withColumnRenamed("body", "value")
        .coalesce(1)
        .write
        .mode("overwrite")
        .text("output_txt_big/")
    )

    print("Done. Output written to output_txt/")
