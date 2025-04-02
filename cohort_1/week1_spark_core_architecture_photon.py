# Databricks notebook source
# Create widgets for parameterization
dbutils.widgets.text("maxPartitionBytes", "256m", "Max Partition Bytes")
dbutils.widgets.text("shufflePartitions", "20", "Shuffle Partitions")

# Read widget values
max_partition_bytes = dbutils.widgets.get("maxPartitionBytes")
shuffle_partitions = dbutils.widgets.get("shufflePartitions")

# Apply configurations dynamically
spark.conf.set("spark.sql.files.maxPartitionBytes", max_partition_bytes)
spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding File Partitioning in Spark
# MAGIC Demo: spark.sql.files.maxPartitionBytes
# MAGIC
# MAGIC Show how this setting affects the number and size of partitions.
# MAGIC
# MAGIC

# COMMAND ----------

spark.conf.set("spark.databricks.io.cache.enabled", "false")

# COMMAND ----------

READ_PATH_FOR_PARQUET_FILES = f"/fake_orders_parquet/"

# COMMAND ----------

input_df = spark.read.parquet(READ_PATH_FOR_PARQUET_FILES)

# COMMAND ----------

input_df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC look at the number of files read and contrast with getNumPartitions()

# COMMAND ----------

display(input_df)

# COMMAND ----------

input_df.rdd.getNumPartitions()

# COMMAND ----------

# spark.conf.set("spark.sql.files.maxPartitionBytes", "10m")  # Example: Set to 10 MB


# input_with_different_max_partition_bytes_df = spark.read.parquet(READ_PATH_FOR_PARQUET_FILES)
# input_with_different_max_partition_bytes_df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicate Pushdown in Spark
# MAGIC Concept: What is predicate pushdown and why it improves performance.
# MAGIC
# MAGIC Demo: Predicate pushdown with Parquet files
# MAGIC
# MAGIC Show filters on columns and how they reduce file/row scanning.
# MAGIC
# MAGIC Use explain(true) to view the physical plan and confirm pushdown.
# MAGIC
# MAGIC Show how many files are actually read with and without filters.
# MAGIC

# COMMAND ----------

display(
  input_df.where("orderDate > '2023-09-01'")
  )

# COMMAND ----------

input_df.where("orderTimestamp > '2023-09-01 00:00:00' ").write.format('noop').mode('overwrite').save()

# COMMAND ----------

# MAGIC %md
# MAGIC Contrast at data filters - rows skipped with and without Photon

# COMMAND ----------

display(
  input_df.where("orderDate > '2023-09-01'").select("orderDate").distinct()
  )

# COMMAND ----------

input_df.where("orderDate > '2023-09-01'").explain(True)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Limitations of Predicate Pushdown
# MAGIC Map data types:
# MAGIC
# MAGIC Show a sample dataset with a MapType column.
# MAGIC
# MAGIC Demonstrate that predicate pushdown does not work on map fields.
# MAGIC
# MAGIC Casts and pushdown:
# MAGIC
# MAGIC If you cast a column (e.g., col("age").cast("int") > 30), predicate pushdown won’t happen.
# MAGIC
# MAGIC Show explain() plan before and after casting. 
# MAGIC
# MAGIC Contrast with and without Photon; look at where the filter is applied

# COMMAND ----------

display(spark.read.parquet(READ_PATH_FOR_PARQUET_FILES).select("customer.userId"))

# COMMAND ----------

spark.read.parquet(READ_PATH_FOR_PARQUET_FILES).where(f"CAST(customer.userId as string)= '80612045'").write.format('noop').mode('overwrite').save()

# COMMAND ----------

# MAGIC %md Show Special Audit Columns
# MAGIC Use spark_partition_id() to visualize how records are split across partitions.
# MAGIC
# MAGIC Use input_file_name() and _metadata columns to examine source file information.
# MAGIC
# MAGIC Show what a no-op transformation looks like (i.e., transformations that don't trigger any computation on their own).

# COMMAND ----------

from pyspark.sql.functions import spark_partition_id

display(
  spark.read.parquet(READ_PATH_FOR_PARQUET_FILES).select(
    '*',
    "_metadata",
    spark_partition_id()
  )
)

# COMMAND ----------

display(
  spark.read.parquet(READ_PATH_FOR_PARQUET_FILES).select(
    '*',
    "_metadata",
    spark_partition_id().alias("partition_id")
  ).select("partition_id").distinct()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark Transformations Quiz
# MAGIC Pop Quiz:
# MAGIC
# MAGIC Is explode() a narrow or wide transformation?
# MAGIC
# MAGIC Provide explanation and a quick demo of the shuffle caused by explode()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to pass parameters to the job
# MAGIC ### Demo how to schedule a job
# MAGIC

# COMMAND ----------



