from __future__ import print_function
"""

Test set accuracy = 0.620119143458

"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, CrossValidator, ParamGridBuilder

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

pipeline_model_path = "output/forest_pipeline_model"

spark = SparkSession.builder.appName("RandomForestClassifier") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

print("Loading Data")
df = spark.read.table("training")

meta_cols = ["HasDetections", "MachineIdentifier"]
selected_cols = ["Platform", "SmartScreen", "EngineVersion", "AvSigVersion", "AppVersion", "CountryIdentifier", "Census_OSBuildNumber", "AVProductStatesIdentifier"]
drop_cols = list(set(df.columns) - set(selected_cols) - set(meta_cols)) 

print("Performing Cleanup")
df = df.drop(*drop_cols)
df.cache()

print("Creating Splits")
train, test = df.randomSplit([0.7, 0.3])

print("Selected Features Count: {0}".format(len(selected_cols)))
print("Selected Features: {0}".format(selected_cols))

print("Building Pipeline")
hasher = FeatureHasher(numFeatures=1024, inputCols=selected_cols, outputCol="features", categoricalCols=selected_cols)
forest = RandomForestClassifier(featuresCol="features", labelCol="HasDetections", predictionCol="prediction", probabilityCol="probability")

pipeline = Pipeline(stages=[hasher, forest])
evaluator = MulticlassClassificationEvaluator(labelCol="HasDetections", predictionCol="prediction", metricName="accuracy")

print("Configuring Validation")
params = ParamGridBuilder() \
			.addGrid(hasher.numFeatures, [1024]) \
			.addGrid(forest.maxDepth, [30]) \
			.addGrid(forest.maxBins, [64]) \
			.addGrid(forest.numTrees, [100]) \
			.build()

#validator = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, parallelism=2)
validator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=3)

print("Fitting -> Training Data")
pipeline_model = validator.fit(train)

print("Fitting -> Test Data")
predictions = pipeline_model.transform(test)
predictions.select("HasDetections", "MachineIdentifier", "probability", "prediction").show(truncate=False)

print("Computing Accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = {0}".format(accuracy))

print("Saving Pipeline Model")
pipeline_model.bestModel.write().overwrite().save(pipeline_model_path)

print("Saving Predictions")
predictions.coalesce(5).write.saveAsTable("forest_predictions", format="parquet", mode="overwrite")