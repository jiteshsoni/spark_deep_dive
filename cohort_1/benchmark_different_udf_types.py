# Databricks notebook source
import time
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf,
    pandas_udf,
    col,
    regexp_replace,
    substring,
    concat,
    lit
)
from pyspark.sql.types import StringType
from pandas import Series

def run_benchmarks(delta_table: str):
    spark = (
        SparkSession
        .builder
        .appName("BenchmarkPhoneFormattingUDFs")
        .getOrCreate()
    )

    # 1️⃣ Read & cache your source Delta table
    df = spark.table(delta_table).repartition(12).cache()
    # materialize cache
    df.write.format("noop").mode("overwrite").save()

    # 2️⃣ Phone‐formatting logic
    def fmt_phone_py(s: str) -> str:
        digits = re.sub(r"\D", "", s or "")[-10:]
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"

    # 2.1 Python UDF (no Arrow)
    py_fmt_phone_noarrow = udf(fmt_phone_py, StringType(), useArrow=False)
    # 2.2 Python UDF (Arrow-optimized)
    py_fmt_phone_arrow   = udf(fmt_phone_py, StringType(), useArrow=True)

    # 2.3 Fully-vectorized Pandas UDF
    @pandas_udf(StringType())
    def pd_fmt_phone(s: Series) -> Series:
        d = s.str.replace(r"\D", "", regex=True).str[-10:]
        return "(" + d.str[:3] + ") " + d.str[3:6] + "-" + d.str[6:]

    # 2.4 Native Spark SQL expression
    def df_sql():
        cleaned   = regexp_replace(col("phone_number"), r"\D", "")
        last10    = substring(cleaned, -10, 10)
        fmt_phone = concat(
            lit("("), substring(last10, 1, 3), lit(") "),
            substring(last10, 4, 3), lit("-"),
            substring(last10, 7, 4)
        ).alias("fmt_phone")
        return df.select(fmt_phone)

    # 3️⃣ Benchmark helper
    def bench(name: str, transformed_df):
        t0 = time.time()
        transformed_df.write.format("noop").mode("overwrite").save()
        print(f"{name:25s} elapsed {time.time() - t0:.3f}s")

    # 4️⃣ Run each variant
    # 4.1 Python UDF without Arrow
    df_py_noarrow = df.withColumn("fmt_phone", py_fmt_phone_noarrow(col("phone_number")))
    bench("Python UDF (no Arrow)", df_py_noarrow)

    # 4.2 Python UDF with Arrow
    df_py_arrow = df.withColumn("fmt_phone", py_fmt_phone_arrow(col("phone_number")))
    bench("Python UDF (Arrow)", df_py_arrow)

    # 4.3 Pandas UDF
    df_pd = df.withColumn("fmt_phone", pd_fmt_phone(col("phone_number")))
    bench("Pandas UDF", df_pd)

    # 4.4 Native SQL
    bench("Native SQL", df_sql())


if __name__ == "__main__":
    run_benchmarks("main.default.fake_customers")


# COMMAND ----------


