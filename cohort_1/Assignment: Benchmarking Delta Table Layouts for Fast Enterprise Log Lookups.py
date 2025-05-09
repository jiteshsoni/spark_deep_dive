# Databricks notebook source
# MAGIC %md
# MAGIC ## Assignment
# MAGIC  https://docs.google.com/document/d/132TrdORxsn57Jg0vpxc7T8z1GKGHO3YQAo8wWGs-xVw/edit?usp=sharing

# COMMAND ----------

# MAGIC %pip install dbldatagen
# MAGIC %pip install Faker

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, floor, lit, from_unixtime, struct, to_json, expr
from datetime import datetime

# Start Spark session
spark = SparkSession.builder.appName("TelemetryLogsWithVariant").getOrCreate()

# Parameters
PARTITIONS = 200
NUM_ROWS = 1_000_000_000  # You can scale this up


NUM_ORGS = 10000

# Schema without event_timestamp or payload
telemetry_schema = StructType([
    StructField("organization_id", StringType(), False),
    StructField("event_type", StringType(), False),
    StructField("browser", StringType(), False),
    StructField("os", StringType(), False),
    StructField("region", StringType(), False),
    StructField("device_id", StringType(), False),
    StructField("referrer", StringType(), False)
])

# Fix device_id by generating integer value correctly
telemetry_spec = (
    dg.DataGenerator(spark, name="telemetry_logs", partitions=PARTITIONS, rows=NUM_ROWS)
    .withSchema(telemetry_schema)
    .withColumnSpec("organization_id", values=[f"org_{i}" for i in range(1, NUM_ORGS + 1)], random=True)
    .withColumnSpec("event_type", values=[
        "login", "logout", "click", "search", "play", "pause", "purchase", "add_to_cart", "remove_from_cart"
    ], random=True)
    .withColumnSpec("browser", values=["Chrome", "Firefox", "Safari", "Edge"], random=True)
    .withColumnSpec("os", values=["Windows", "macOS", "Linux", "iOS", "Android"], random=True)
    .withColumnSpec("region", values=["NA", "EU", "ASIA", "SA", "AF"], random=True)
    .withColumnSpec("device_id", minValue=1000000, maxValue=9999999, random=True)
    .withColumnSpec("referrer", values=["ad", "search", "direct", "email"], random=True)
)

# Build DataFrame
telemetry_df = telemetry_spec.build(withStreaming=False)

# Add proper UUIDs
telemetry_df = telemetry_df.withColumn("session_id", expr("uuid()"))

# Add random event_timestamp in 2024
seconds_in_2024 = 90 * 24 * 60 * 60
telemetry_df = telemetry_df.withColumn(
    "event_timestamp",
    from_unixtime(
        floor(
            lit(datetime(2024, 1, 1).timestamp()) + rand() * seconds_in_2024
        ).cast("long")
    )
)

# Construct payload as JSON string
telemetry_df = telemetry_df.withColumn(
    "event_payload_json",
    to_json(struct("browser", "os", "region", "device_id", "session_id", "referrer"))
)

# Convert to VARIANT using parse_json (Databricks Delta 3.0+)
telemetry_df = telemetry_df.withColumn(
    "event_payload", expr("parse_json(event_payload_json)")
).drop("event_payload_json")

# Clean columns using select * except [...]
exclude_cols = {"browser", "os", "region", "device_id", "session_id", "referrer", "event_payload_json"}
telemetry_df = telemetry_df.select(*[c for c in telemetry_df.columns if c not in exclude_cols])


#display(telemetry_df)

# COMMAND ----------

telemetry_df.printSchema()


# COMMAND ----------

# Optional: Save as Delta Table
TARGET_TABLE = "soni.default.telemetry_logs"

# COMMAND ----------


telemetry_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)

print(f"Generated and saved to table: {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Liquid Clustered table
# MAGIC #### ✅ Precondition: You must be on Databricks Runtime 13.3+ with Delta Lake 3.0+ and Unity Catalog or Catalog-enabled metastore.

# COMMAND ----------

# Read the original Delta table
source_df = spark.read.table(TARGET_TABLE)

# Create new Liquid Clustered table
(
    source_df.write
    .mode("overwrite")  # or 'append' if not overwriting \
    .clusterBy("organization_id", "event_type", "event_timestamp") \
    .saveAsTable("soni.default.telemetry_logs_variant_lc")  # new table name
)

print("✅ Liquid Clustered table created as telemetry_logs_variant_lc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a  Partitioned and Zorder Table

# COMMAND ----------

# Read the previous table
telemetry_df = spark.table(TARGET_TABLE)

# Create a partitioned table on organization_id and ZORDER it by event_type and event_timestamp
telemetry_df.write \
    .partitionBy("organization_id") \
    .mode("overwrite") \
    .saveAsTable("soni.default.telemetry_logs_partitioned")



# COMMAND ----------

# Optimize the table with ZORDER
spark.sql("""
    OPTIMIZE soni.default.telemetry_logs_partitioned
    ZORDER BY (event_type, event_timestamp)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Design Your Own Partitioning Strategy
# MAGIC Consider experimenting with partitioning on two different columns for improved filtering.
# MAGIC
# MAGIC You can also create a derived partition column by transforming existing data — for example:
# MAGIC
# MAGIC Applying a hash function on a string column
# MAGIC
# MAGIC Using only the first or last few characters of a column value
# MAGIC
# MAGIC Truncating or prefixing values to group similar entries

# COMMAND ----------

# FILL IT HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmarking on SQL Warehouse with the smallest warehouse is 2x-small
# MAGIC ### Tips for benchmarking

# COMMAND ----------

# MAGIC %md 
# MAGIC * Generate random organization_id values from a pool of only 1,000 sequential IDs to simulate repeated access patterns.
# MAGIC
# MAGIC * Use random 15-minute time windows when generating event_timestamp filters. This helps bypass DBSQL result caching, which can otherwise skew performance metrics.
# MAGIC
# MAGIC * Vary the query filters: don’t always include all three columns (organization_id, event_type, event_timestamp). Try combinations of two or more observe how that impacts performance.
# MAGIC
# MAGIC
