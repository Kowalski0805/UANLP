import time

from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F

# Modified speed test for JVM, relying on UDF to process the text via Java (Jmorphy or Morfologik)
if __name__ == "__main__":
    spark = (
        SparkSession
            .builder
            .appName("speedTest")
            .config("spark.rapids.sql.enabled", "false")
            .getOrCreate()
    )

    spark.udf.registerJavaFunction("morph_analyzer", "org.example.MorphAnalyzerUDF")
    spark.udf.registerJavaFunction("morfologik", "org.example.MorfologikUDF")

    spark.catalog.clearCache()
    df = spark.read.parquet("data_parquet/")

    df = df \
        .withColumn("body_vec", F.expr("morfologik(body)")) \
        .withColumn("title_vec", F.expr("morfologik(title)"))

    start = time.time()
    df.write.mode("overwrite").parquet("benchmark_output/")
    end = time.time()
    print(f"⏱ Time taken: {end - start:.2f} seconds")
    df.explain()
    print(df.tail(20))

    print("💤 Sleeping to keep Spark UI alive at http://localhost:4040")
    time.sleep(99999)  # or however long you want
