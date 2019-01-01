from __future__ import print_function
"""
Bayes Rule:

P(Malware | X) = P(X | Malware) P(Malware) / P(X)

Where X = 83 column values in each dataset:
"""
import pickle

from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F

pipeline_model_path = "output/naive_bayes_pipeline_model"

#selected_cols = ["Platform", "AvSigVersion", "AVProductStatesIdentifier", "CountryIdentifier", "SMode", "Wdft_RegionIdentifier", "Census_OSVersion", "OsBuildLab", "Census_IsTouchEnabled", "Census_IsPenCapable", "Census_IsSecureBootEnabled"]
#selected_cols = ["Platform", "AvSigVersion", "AVProductStatesIdentifier", "CountryIdentifier", "Census_OSVersion", "Derived_Firmware", "Derived_Processor", "Derived_Installer", "Derived_AV", "Derived_Grouping", "Census_IsTouchEnabled", "Census_IsSecureBootEnabled"]

df = spark.read.load("../datasets/train.csv", format="csv", sep=",", inferSchema="true", header="true")
df.cache()

exclude_cols = ["HasDetections", "MachineIdentifier", "DefaultBrowsersIdentifier", "OrganizationIdentifier", "GeoNameIdentifier", "LocaleEnglishNameIdentifier", "Census_ProcessorCoreCount", "Census_ProcessorClass", "Census_PrimaryDiskTotalCapacity", "Census_PrimaryDiskTypeName", "Census_SystemVolumeTotalCapacity", "Census_HasOpticalDiskDrive", "Census_TotalPhysicalRAM", "Census_InternalPrimaryDiagonalDisplaySizeInInches", "Census_InternalPrimaryDisplayResolutionHorizontal", "Census_InternalBatteryType", "Census_InternalBatteryNumberOfCharges"]
selected_cols = list(set(df.columns) - set(exclude_cols))

print("Selected Features Count: {0}".format(len(selected_cols)))
print("Selected Features: {0}".format(selected_cols))
print("Selected Features: {0}".format(len(selected_cols)))

print("Creating Splits")
train, test = df.randomSplit([0.7, 0.3])

print("Building Pipeline")
hasher = FeatureHasher(inputCols=selected_cols, outputCol="features", categoricalCols=selected_cols)
evaluator = MulticlassClassificationEvaluator(labelCol="HasDetections", predictionCol="prediction", metricName="accuracy")
stages = []
stages.append(hasher)
stages.append(DecisionTreeClassifier(featuresCol='features', labelCol='HasDetections', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=2048, cacheNodeIds=False, checkpointInterval=10, impurity='gini', seed=None))
pipeline = Pipeline(stages=stages)

print("Configuring CrossValidation")
params = ParamGridBuilder() \
			.addGrid(hasher.numFeatures, [32768, 65536]) \
			.build()

validator = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=params,
                          evaluator=evaluator,
                          numFolds=5)

print("Fitting -> Training Data")
pipeline_model = validator.fit(train)

print("Fitting -> Test Data")
predictions = pipeline_model.transform(test)
predictions.select("HasDetections", "MachineIdentifier", "probability", "prediction").show(truncate=False)

print("FeatureHasher.numFeatures = {0}".format(stages[0].getNumFeatures()))
stages[0].getNumFeatures()

print("Computing Multiclass Accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = {0}".format(accuracy))

print("Saving Pipeline Model")
pipeline_model.bestModel.write().overwrite().save(pipeline_model_path)

#print("Saving Predictions")
#predictions.write.saveAsTable("naive_bayes_predictions", format="parquet", mode="overwrite", path="output/tables/naive_bayes/predictions")




