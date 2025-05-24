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
    spark = (
        SparkSession.builder
            .appName("speedTest-RAPIDS")
            # .master("local[*]")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
    )

    df_parquet = spark.read.parquet("data_parquet")
    replicated_df = df_parquet

    for _ in range(16):  # 3 doublings = 8x data
        replicated_df = replicated_df.union(df_parquet)

    replicated_df.write.mode("overwrite").parquet("data_parquet_big")


