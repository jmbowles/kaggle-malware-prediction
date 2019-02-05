from __future__ import print_function
"""

https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
https://mlwave.com/kaggle-ensembling-guide/

Test set accuracy = 0.640211713761

"""
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder \
	.appName("EnsembleEval") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()


def transform(df, model_path, prediction_column):
	pipeline_model = PipelineModel.load(model_path)
	predictions = pipeline_model.transform(df)
	predictions = predictions.drop(*["features", "rawPrediction", "probability", "categorical_features", "continuous_features", "continuous_vector"])
	predictions = predictions.withColumnRenamed("prediction", prediction_column)
	return predictions

def evaluate(df, prediction_column):
	print("Computing Accuracy for {0}".format(prediction_column))
	evaluator = MulticlassClassificationEvaluator(labelCol="HasDetections", predictionCol=prediction_column, metricName="accuracy")
	accuracy = evaluator.evaluate(df)
	return accuracy


print("Loading and Caching Data")
df = spark.read.table("training")
train, test = df.randomSplit([0.5, 0.5])
df = test
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
predictions = predictions.withColumn("majority_prediction", F.when(predictions.majority_sum >= 3.0, F.lit(1)).otherwise(F.lit(0)).cast("double"))

print("Performing Evaluation")
accuracy = evaluate(predictions, "majority_prediction")
print("Test set accuracy = {0}".format(accuracy))