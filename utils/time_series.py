from __future__ import print_function
"""
Many thanks to David Kaleko:

http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
"""
import numpy as np
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


spark = SparkSession.builder \
	.appName("TimeSeries") \
	.config("spark.sql.execution.arrow.enabled", "true") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

def to_pandas(numpy_data, columns):
	return pd.concat([pd.DataFrame([(k, v.isoformat())], columns=columns) for _, (k, v) in enumerate(numpy_data.items())], ignore_index=True)

def add_cyclical_features(df, source_column, target_column):
	df = df.withColumn(target_column, F.from_utc_timestamp(F.col(source_column), "UTC"))
	hour = target_column + "_hour"
	weekday = target_column + "_weekday"
	month = target_column + "_month"
	year = target_column + "_year"
	df = df.withColumn(hour, F.hour(F.col(target_column)))
	df = df.withColumn(weekday, F.dayofweek(F.col(target_column))-1)
	df = df.withColumn(month, F.month(F.col(target_column))-1)
	df = df.withColumn("DerivedTime_cos_" + hour, F.cos(F.col(hour) * (2.0 * np.pi / 24)) + 1)
	df = df.withColumn("DerivedTime_sin_" + hour, F.sin(F.col(hour) * (2.0 * np.pi / 24)) + 1)
	df = df.withColumn("DerivedTime_cos_" + weekday, F.cos(F.col(weekday) * (2.0 * np.pi / 7)) + 1)
	df = df.withColumn("DerivedTime_sin_" + weekday, F.sin(F.col(weekday) * (2.0 * np.pi / 7)) + 1)
	df = df.withColumn("DerivedTime_cos_" + month, F.cos(F.col(month) * (2.0 * np.pi / 12)) + 1)
	df = df.withColumn("DerivedTime_sin_" + month, F.sin(F.col(month) * (2.0 * np.pi / 12)) + 1)
	df = df.withColumn("DerivedTime_" + year, F.year(F.col(target_column)))
	drop_cols = [source_column, hour, weekday, month]
	df = df.drop(*drop_cols)
	return df

schema_signatures = StructType([StructField("AvSigVersion", StringType(), False), StructField("AvSigVersionDate", StringType(), False)])
schema_os = StructType([StructField("Census_OSVersion", StringType(), False), StructField("OSVersionDate", StringType(), False)])

print("Loading Data")
numpy_signatures = np.load('../datasets/AvSigVersionTimestamps.npy')[()]
numpy_os = np.load('../datasets/OSVersionTimestamps.npy')[()]

print("Converting Data from Numpy to Pandas")
pdf_signatures = to_pandas(numpy_signatures, ["AvSigVersion", "AvSigVersionDate"])
pdf_os = to_pandas(numpy_os, ["Census_OSVersion", "OSVersionDate"])

print("Creating Spark Dataframes")
df_signatures = spark.createDataFrame(pdf_signatures, schema_signatures)
df_signatures = add_cyclical_features(df_signatures, "AvSigVersionDate", "AvSigVersionUTC")

df_os = spark.createDataFrame(pdf_os, schema_os)
df_os = add_cyclical_features(df_os, "OSVersionDate", "OSVersionUTC")

df_signatures.coalesce(1).write.parquet(path="../datasets/avsigversion_timestamps", mode="overwrite")
df_os.coalesce(1).write.parquet(path="../datasets/osversion_timestamps", mode="overwrite")