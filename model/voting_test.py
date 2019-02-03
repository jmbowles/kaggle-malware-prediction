from __future__ import print_function
"""

https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
https://mlwave.com/kaggle-ensembling-guide/

"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder \
	.appName("VotingTest") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

df = spark.createDataFrame([(0.0,0.0,0.0,0.0,0.0,0.0), 
							(1.0,0.0,0.0,0.0,0.0,0.0),
							(1.0,1.0,0.0,0.0,0.0,0.0),
							(1.0,1.0,1.0,0.0,0.0,0.0), 
							(1.0,1.0,1.0,1.0,0.0,0.0),
							(1.0,1.0,1.0,1.0,1.0,0.0),
							(1.0,1.0,1.0,1.0,1.0,1.0)],['m1','m2','m3','m4','m5','m6'])

vote = df.withColumn("majority_sum", df.m1 + df.m2 + df.m3 + df.m4 + df.m5 + df.m6)
vote = vote.withColumn("majority_vote", F.when(vote.majority_sum >= 3.0, F.lit(1)).otherwise(F.lit(0)))
vote = vote.withColumn("HasDetections", vote.majority_vote)
vote.show()