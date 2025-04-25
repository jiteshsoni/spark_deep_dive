# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmarking & Catalyst-Blindness Demo
# MAGIC
# MAGIC In this notebook we will:
# MAGIC
# MAGIC 1. Read & cache a Delta table  
# MAGIC 2. Register three UDF variants + native SQL  
# MAGIC 3. **Benchmark** each variant side by side  
# MAGIC 4. **Walk through** each variant’s physical plan (with `explain(true)`)  
# MAGIC 5. Optionally, add a filter to show lost predicate push-down 

# COMMAND ----------

# Cell 1: Imports
import time, re
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf, pandas_udf, col,
    regexp_replace, substring, concat, lit
)
from pyspark.sql.types import StringType
from pandas import Series

# COMMAND ----------

# Cell 2: Spark setup & cache
spark = (SparkSession
         .builder
         .appName("Demo_Optimizer_Blindness")
         .getOrCreate())

# Replace with your table
table_name = "main.default.fake_customers"

df = (spark.table(table_name)
      .repartition(12)
      .cache())

# materialize cache
df.write.format("noop").mode("overwrite").save()

print(f"Loaded & cached `{table_name}` → {df.count():,} rows")


# COMMAND ----------

# 2.4 Native-SQL formatter via withColumn
phone_clean = regexp_replace(col("phone_number"), r"\D", "")
last10     = substring(phone_clean, -10, 10)
fmt_expr   = concat(
    lit("("), substring(last10,1,3), lit(") "),
    substring(last10,4,3), lit("-"),
    substring(last10,7,4)
)

native_with_column = df.withColumn("fmt_phone", fmt_expr)

# COMMAND ----------

# Cell 3: Register UDFs
def fmt_phone_py(s: str) -> str:
    digits = re.sub(r"\D", "", s or "")[-10:]
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"

py_udf_noarrow = udf(fmt_phone_py, StringType(), useArrow=False)
py_udf_arrow   = udf(fmt_phone_py, StringType(), useArrow=True)

@pandas_udf(StringType())
def pd_udf(s: Series) -> Series:
    d = s.str.replace(r"\D", "", regex=True).str[-10:]
    return "(" + d.str[:3] + ") " + d.str[3:6] + "-" + d.str[6:]

print("Registered: py_udf_noarrow, py_udf_arrow, pd_udf")


# COMMAND ----------

# # Cell 4: Native-SQL phone formatting helper
# def df_sql(df):
#     c = regexp_replace(col("phone_number"), r"\D", "")
#     l = substring(c, -10, 10)
#     fmt = concat(
#         lit("("), substring(l,1,3), lit(") "),
#         substring(l,4,3), lit("-"),
#         substring(l,7,4)
#     ).alias("fmt_phone")
#     return df.select(fmt)


# print("Defined df_sql(df)")


# COMMAND ----------

# Cell 5: Build all four pipelines
pipelines = {
    "Native SQL (withColumn)": native_with_column,
    "Python UDF (no Arrow)"   : df.withColumn("fmt_phone", py_udf_noarrow(col("phone_number"))),
    "Python UDF (Arrow)"      : df.withColumn("fmt_phone", py_udf_arrow(col("phone_number"))),
    "Pandas UDF"              : df.withColumn("fmt_phone", pd_udf(col("phone_number"))),
}
print("Pipelines ready:", list(pipelines.keys()))


# COMMAND ----------

# Cell 6: Benchmark all variants
print("\n=== BENCHMARK RESULTS ===")
print(f"{'Variant':30s} | {'Seconds':>7s}")
print("-"*42)
for name, p_df in pipelines.items():
    t0 = time.time()
    p_df.write.format("noop").mode("overwrite").save()
    print(f"{name:30s} | {time.time()-t0:7.3f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Walk through the plans
# MAGIC
# MAGIC Below we’ll examine each variant’s **physical plan**.  For each:
# MAGIC
# MAGIC - **Native SQL**: look for one big `WholeStageCodegen` block  
# MAGIC - **UDF variants**: look for a `PythonExec` (or `PythonRunner`) stage, and notice **no** single WholeStage wrapping your logic  
# MAGIC

# COMMAND ----------

for name, p_df in pipelines.items():
    print(f"\n\n\n\n\n --- PLAN: {name} ---")
    p_df.explain(True)

# COMMAND ----------

sql_f = pipelines["Native SQL"].filter(col("fmt_phone").startswith("(415)"))
sql_f.write.format("noop").mode("overwrite").save()

# COMMAND ----------

udf_f = pipelines["Python UDF (no Arrow)"].filter(col("fmt_phone").startswith("(415)"))
udf_f.write.format("noop").mode("overwrite").save()

# COMMAND ----------

sql_f = pipelines["Native SQL"].filter(col("fmt_phone").startswith("(415)"))


print("\n--- Native SQL + Filter ---")
sql_f.explain(True)

print("\n--- UDF (no Arrow) + Filter ---")
udf_f.explain(True)


# COMMAND ----------

# Cell 7: Explain Native SQL
print("\n--- Native SQL Plan ---")
pipelines["Native SQL"].explain(True)


# COMMAND ----------

# Cell 8: Explain Python UDF (no Arrow)
print("\n--- Python UDF (no Arrow) Plan ---")
pipelines["Python UDF (no Arrow)"].explain(True)


# COMMAND ----------

print("SQL optimized plan:")
print(pipelines["SQL"].queryExecution.optimizedPlan.treeString())

print("UDF optimized plan:")
print(pipelines["UDF"].queryExecution.optimizedPlan.treeString())


# COMMAND ----------

# MAGIC %sql
# MAGIC -- run in a SQL cell in Databricks
# MAGIC ANALYZE TABLE main.default.fake_customers COMPUTE STATISTICS FOR COLUMNS phone_number;
# MAGIC

# COMMAND ----------

tbl = spark.table("main.default.fake_customers")
stats = tbl._jdf.queryExecution().optimizedPlan().stats()
print("Base table stats:", stats)    # should show a defined rowCount & sizeInBytes


# COMMAND ----------

native_df = df_sql(tbl)
udf_df    = tbl.withColumn("fmt_phone", py_udf_noarrow(col("phone_number")))


# COMMAND ----------

native_stats = native_df._jdf.queryExecution().optimizedPlan().stats()
udf_stats    = udf_df   ._jdf.queryExecution().optimizedPlan().stats()

print("Native pipeline stats:", native_stats)   # rowCount, sizeInBytes set
print("UDF pipeline   stats:", udf_stats)      # rowCount=None, sizeInBytes=0 (unknown)


# COMMAND ----------


