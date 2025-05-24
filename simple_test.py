from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SimpleGPUJob") \
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
    .config("spark.rapids.sql.enabled", "true") \
    .config("spark.executor.resource.gpu.amount", "1") \
    .config("spark.task.resource.gpu.amount", "1") \
    .getOrCreate()

data = spark.range(100000).selectExpr("id", "id * 2 as value")
data.show()
