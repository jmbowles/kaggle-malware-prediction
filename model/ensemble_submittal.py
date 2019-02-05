from __future__ import print_function
"""


"""
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

csv_path = "output/submit/submittal_9.csv"

spark = SparkSession.builder \
	.appName("EnsembleSubmittal") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.enableHiveSupport() \
	.getOrCreate()


def transform(df, model_path, prediction_column):
	pipeline_model = PipelineModel.load(model_path)
	predictions = pipeline_model.transform(df)
	predictions = predictions.drop(*["features", "rawPrediction", "probability", "categorical_features", "continuous_features", "continuous_vector"])
	predictions = predictions.withColumnRenamed("prediction", prediction_column)
	return predictions


print("Loading and Caching Data")
df = spark.read.table("test")
df.cache()

print("Transforming Model Predictions")
predictions = transform(df, "output/logistic_pipeline_model", "logistic_prediction")
predictions = transform(predictions, "output/naive_bayes_pipeline_model", "bayes_prediction")
predictions = transform(predictions, "output/gradient_boosted_pipeline_model", "boosted_prediction")
predictions = transform(predictions, "output/perceptron_pipeline_model", "perceptron_prediction")
predictions = transform(predictions, "output/forest_pipeline_model", "forest_prediction")

majority_sum = (predictions.logistic_prediction + predictions.bayes_prediction + predictions.boosted_prediction + 
				predictions.perceptron_prediction + predictions.forest_prediction)

predictions = predictions.withColumn("majority_sum", majority_sum)
predictions = predictions.withColumn("HasDetections", F.when(predictions.majority_sum >= 3.0, F.lit(1)).otherwise(F.lit(0)).cast("double"))
submittal = predictions.select("MachineIdentifier", "HasDetections")

print("Creating CSV for Submittal")
# Yet another workaround to write to a CSV file
submittal.coalesce(1).toPandas().to_csv(csv_path, header=True, index=False)

print("Total rows written to file: {0}".format(submittal.count()))