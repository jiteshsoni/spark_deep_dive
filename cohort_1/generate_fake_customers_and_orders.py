# Databricks notebook source
# MAGIC %pip install dbldatagen
# MAGIC %pip install Faker

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Initialization
# MAGIC
# MAGIC - Import necessary libraries: `dbldatagen` for data generation, `uuid`, `pandas`, `pyspark` for Spark operations, and `Faker` for generating fake data.
# MAGIC - Create a Spark session.
# MAGIC - Initialize the Faker library for generating realistic fake data.
# MAGIC
# MAGIC ## Parameters
# MAGIC
# MAGIC - Define constants for the number of partitions, the number of customers (100,000), rows per second (placeholder for streaming), and the target Delta table name.
# MAGIC
# MAGIC ## Generate Initial DataFrame
# MAGIC
# MAGIC - Define a schema with only a `customer_id` field.
# MAGIC - Use `dbldatagen` to create a DataFrame with `customer_id` values ranging from 1 to 100,000.
# MAGIC - Limit the DataFrame to 100,000 rows.
# MAGIC
# MAGIC ## Define a Vectorized UDF
# MAGIC
# MAGIC - Define the schema for the UDF's return value, which includes fields like name, email, address, city, state, zip code, and phone number.
# MAGIC - Create a Pandas UDF (`vectorized_gen_customer_info`) that generates fake customer details for each `customer_id` using the Faker library.
# MAGIC
# MAGIC ## Enrich the DataFrame
# MAGIC
# MAGIC - Apply the UDF to the `customer_id` column to generate additional customer details.
# MAGIC - Expand the resulting struct column into individual columns.
# MAGIC
# MAGIC ## Write to Delta Table
# MAGIC
# MAGIC - Write the enriched DataFrame to a Delta table in overwrite mode.
# MAGIC
# MAGIC ## Print Confirmation
# MAGIC
# MAGIC - Print a message confirming that the DataFrame has been generated and saved to the specified Delta table.
# MAGIC
# MAGIC This script effectively combines data generation, enrichment with realistic details, and storage in a Delta table, making it useful for testing and development purposes.

# COMMAND ----------

import dbldatagen as dg
import uuid
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.sql.functions import pandas_udf
from pyspark.sql import SparkSession
from faker import Faker

# Create a Spark session (if not already created)
spark = SparkSession.builder.appName("CustomersDataGen").getOrCreate()

# Initialize Faker
fake = Faker()

# -------------------------------
# Parameters
# -------------------------------
PARTITIONS = 50
NUM_CUSTOMERS = 500_000    # Targeting 500k rows
TARGET_TABLE = "main.default.fake_customers"  # Delta table name

# -------------------------------
# 1. Generate initial DataFrame with customer_id only using dbldatagen
# -------------------------------
# Define a basic schema that only contains customer_id.
customers_schema = StructType([
    StructField("customer_id", LongType(), False)
])

customers_dataspec = (
    dg.DataGenerator(spark, name="customers_data", partitions=PARTITIONS)
    .withSchema(customers_schema)
    .withColumnSpec("customer_id", minValue=1, maxValue=NUM_CUSTOMERS, random=True)
)

# Build a static (batch) DataFrame.
customers_df = customers_dataspec.build(withStreaming=False)

# -------------------------------
# 2. Define a vectorized (Pandas) UDF to enrich the DataFrame with realistic customer details
# -------------------------------
# Define the return schema for the vectorized UDF (excluding customer_id).
udf_return_schema = StructType([
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("addressLine1", StringType(), True),
    StructField("addressLine2", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip_code", StringType(), True),
    StructField("phone_number", StringType(), True)
])

@pandas_udf(udf_return_schema)
def vectorized_gen_customer_info(customer_ids: pd.Series) -> pd.DataFrame:
    n = len(customer_ids)
    # Generate lists for each field using Faker
    names    = [fake.name() for _ in range(n)]
    emails   = [fake.email() for _ in range(n)]
    addr1    = [fake.street_address() for _ in range(n)]
    addr2    = [fake.secondary_address() if fake.boolean(chance_of_getting_true=50) else "" for _ in range(n)]
    cities   = [fake.city() for _ in range(n)]
    states   = [fake.state_abbr() for _ in range(n)]
    zip_codes= [fake.zipcode() for _ in range(n)]
    phones   = [fake.phone_number() for _ in range(n)]
    
    return pd.DataFrame({
        "name": names,
        "email": emails,
        "addressLine1": addr1,
        "addressLine2": addr2,
        "city": cities,
        "state": states,
        "zip_code": zip_codes,
        "phone_number": phones
    })

# -------------------------------
# 3. Enrich the DataFrame and Write as a Delta Table
# -------------------------------
# Apply the vectorized UDF on the "customer_id" column (used only to determine the batch size).
customers_df = customers_df.withColumn("info", vectorized_gen_customer_info(customers_df.customer_id))
# Expand the struct column "info" to individual columns.
customers_df = customers_df.drop_duplicates(["customer_id"]).select("customer_id", "info.*")

# Write the final DataFrame as a Delta table.
customers_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)

print("Customers DataFrame with 100k rows generated and saved to table:", TARGET_TABLE)


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1), COUNT(disTINCT customer_id)
# MAGIC FROM main.default.fake_customers

# COMMAND ----------

import dbldatagen as dg
import uuid
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, IntegerType, DoubleType
from pyspark.sql import SparkSession

# Create a Spark session.
spark = SparkSession.builder.appName("OrdersDataGen").getOrCreate()

# -------------------------------
# Define the orders schema with at least 10 columns.
# -------------------------------
orders_schema = StructType([
    StructField("order_id", LongType(), False),
    StructField("order_timestamp", TimestampType(), False),
    StructField("customer_id", IntegerType(), False),    # Must be between 1 and 1,000,000.
    StructField("order_amount", DoubleType(), False),
    StructField("product_category", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("order_status", StringType(), True),
    StructField("product_id", IntegerType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("unit_price", DoubleType(), False)
])

# -------------------------------
# Use dbldatagen to generate a base DataFrame with basic columns.
# -------------------------------
# Adjust PARTITIONS based on your cluster's capacity.
PARTITIONS = 100  
# We want a total of 1 billion rows.
NUM_ORDERS = 2_000_000_000

orders_dataspec = (
    dg.DataGenerator(spark, name="orders_data", partitions=PARTITIONS, rows=NUM_ORDERS)
    .withSchema(orders_schema)
    # Generate order_id from 1 to a large number.
    .withColumnSpec("order_id", minValue=1, maxValue=10**12, random=True)
    # Generate an order timestamp between two dates.
    .withColumnSpec("order_timestamp", begin="2023-01-01 00:00:00", end="2023-12-31 23:59:59", random=True)
    # Customer ID between 1 and 1,000,000.
    .withColumnSpec("customer_id", minValue=1, maxValue=NUM_CUSTOMERS, random=True)
    # Order amount between $5 and $5000.
    .withColumnSpec("order_amount", minValue=5.0, maxValue=5000.0, random=True)
    # Product category chosen from a list.
    .withColumnSpec("product_category", values=["Electronics", "Clothing", "Home", "Books", "Toys"], random=True)
    # Payment method chosen from a list.
    .withColumnSpec("payment_method", values=["Credit Card", "Debit Card", "PayPal", "Wire Transfer"], random=True)
    # Order status chosen from a list.
    .withColumnSpec("order_status", values=["Completed", "Pending", "Cancelled"], random=True)
    # Product ID between 1 and 1,000,000,000.
    .withColumnSpec("product_id", minValue=1, maxValue=100_000, random=True)
    # Quantity between 1 and 10.
    .withColumnSpec("quantity", minValue=1, maxValue=10, random=True)
    # Unit price between $5 and $500.
    .withColumnSpec("unit_price", minValue=5.0, maxValue=500.0, random=True)
)

# -------------------------------
# Build the static DataFrame.
# -------------------------------
# In batch mode, we cannot pass a count parameter. Instead, we build the DataFrame and limit it.
orders_df = orders_dataspec.build(withStreaming=False, )

# -------------------------------
# Write the final DataFrame as a Delta table.
# -------------------------------
# This will create or overwrite the Delta table with the generated orders.
TARGET_TABLE = "main.default.fake_orders"
orders_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)

print("Orders DataFrame with 1 billion rows generated and saved to table:", TARGET_TABLE)


# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1), COUNT(DISTINCT order_id)
# MAGIC from main.default.fake_orders
