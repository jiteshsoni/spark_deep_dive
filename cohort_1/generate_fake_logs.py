# Databricks notebook source
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
seconds_in_2024 = 366 * 24 * 60 * 60
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
telemetry_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)

print(f"Generated and saved to table: {TARGET_TABLE}")

# COMMAND ----------


