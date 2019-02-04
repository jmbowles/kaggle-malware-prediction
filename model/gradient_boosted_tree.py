from __future__ import print_function
"""

Test set accuracy = 0.632725844292

"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

pipeline_model_path = "output/gradient_boosted_pipeline_model"

spark = SparkSession.builder.appName("GradientBootedTree") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

print("Loading and Caching Data")
df = spark.read.table("training")

continuous_cols = ["Census_OSBuildNumber", "AVProductStatesIdentifier", "AVProductsInstalled", "OsBuild", "OsSuite", "IeVerIdentifier", 
					"Census_ProcessorCoreCount", "Census_TotalPhysicalRAM", "Census_InternalBatteryNumberOfCharges", "Census_OSBuildRevision"] 

derived_cols = [col for col in df.columns if col.startswith("Derived_")]

exclude_cols = ["HasDetections", "MachineIdentifier", "AvSigVersion", "AppVersion", "EngineVersion", "OsVer", "Census_OSVersion", "Census_FirmwareManufacturerIdentifier", "Census_FirmwareVersionIdentifier", 
				"Census_SystemVolumeTotalCapacity", "Census_PrimaryDiskTotalCapacity", "Census_InternalPrimaryDisplayResolutionHorizontal", "Census_InternalPrimaryDisplayResolutionVertical", "CityIdentifier", 
				"LocaleEnglishNameIdentifier", "DefaultBrowsersIdentifier", "LocaleEnglishNameIdentifier", 
				"Census_ProcessorClass", "Census_InternalBatteryType", "Census_DeviceFamily", "OsPlatformSubRelease", "Census_OSBranch"]

continuous_cols = continuous_cols + derived_cols
categorical_cols = list(set(df.columns) - set(exclude_cols) - set(continuous_cols))
feature_cols = ["categorical_features", "continuous_features"]

meta_cols = ["HasDetections", "MachineIdentifier"]
drop_cols = list(set(df.columns) - set(meta_cols) - set(continuous_cols) - set(categorical_cols) - set(feature_cols) ) 

df = df.drop(*drop_cols)
df.cache()

print("Creating Splits")
train, test = df.randomSplit([0.7, 0.3])

print("Selected Features Count: {0}".format(len(feature_cols)))
print("Selected Features: {0}".format(feature_cols))

print("Building Pipeline")
categorical_hasher = FeatureHasher(inputCols=categorical_cols, outputCol="categorical_features", categoricalCols=categorical_cols)
continuous_vector = VectorAssembler(inputCols=continuous_cols, outputCol="continuous_vector")
scaler = MinMaxScaler(min=0.0, max=1.0, inputCol=continuous_vector.getOutputCol(), outputCol="continuous_features")
features = VectorAssembler(inputCols=feature_cols, outputCol="features")
boosted = GBTClassifier(featuresCol="features", labelCol="HasDetections", predictionCol="prediction")
pipeline = Pipeline(stages=[categorical_hasher, continuous_vector, scaler, features, boosted])
evaluator = MulticlassClassificationEvaluator(labelCol="HasDetections", predictionCol="prediction", metricName="accuracy")

print("Configuring CrossValidation")
params = ParamGridBuilder() \
			.addGrid(categorical_hasher.numFeatures, [512]) \
			.build()

validator = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=params,
                          evaluator=evaluator,
                          numFolds=3)

print("Fitting -> Training Data")
pipeline_model = validator.fit(train)

print("Fitting -> Test Data")
predictions = pipeline_model.transform(test)
predictions.select("HasDetections", "MachineIdentifier", "prediction").show(truncate=False)

print("Computing Multiclass Accuracy")
evaluator = MulticlassClassificationEvaluator(labelCol="HasDetections", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

print("Saving Pipeline Model")
pipeline_model.bestModel.write().overwrite().save(pipeline_model_path)

print("Saving Predictions")
predictions.coalesce(5).write.saveAsTable("gradient_boosted_predictions", format="parquet", mode="overwrite")