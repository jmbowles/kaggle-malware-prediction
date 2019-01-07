from __future__ import print_function

import pickle

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import ChiSqSelector, ChiSqSelectorModel, StringIndexer, VectorAssembler


pipeline_model_path = "output/chi_square"
feature_path = "output/features.pkl"

model = None
feature_cols = None
selector = None

try:
	print("Loading Pipeline")
	model = PipelineModel.load(pipeline_model_path)

	with open(feature_path, "rb") as f:
		feature_cols = pickle.load(f)
except:
	print("INFO: Output path missing")


if not model:

	print("Loading and caching data")
	df = spark.read.load("../datasets/train.csv", format="csv", sep=",", inferSchema="true", header="true")
	df.cache()
	
	categorical_cols = [col for col, dtype in df.dtypes if dtype=="string" or "Is" in col]
	label_col = ["HasDetections"]
	meta_cols = ["MachineIdentifier"]
	feature_cols = list(set(categorical_cols) - set(label_col) - set(meta_cols))
	indexed_cols = []
	stages = []
	
	print("Hashing features")
	for feature in feature_cols:
		indexed = feature + "_" + "indexed"
		indexed_cols.append(indexed)
		indexer = StringIndexer(inputCol=feature, outputCol=indexed, handleInvalid="keep", stringOrderType="frequencyDesc")
		stages.append(indexer)
		
	stages.append(VectorAssembler(inputCols=indexed_cols, outputCol="features", handleInvalid="keep"))
	stages.append(ChiSqSelector(numTopFeatures=20, labelCol="HasDetections", featuresCol="features", outputCol="selectedFeatures"))
	
	print("Performing model fitting")
	pipeline = Pipeline(stages=stages)
	model = pipeline.fit(df)
	df_features = model.transform(df)
	df_features.select("features", "selectedFeatures").show()

	print("Saving Pipeline Model")
	model.write().overwrite().save(pipeline_model_path)

	with open(feature_path, "wb") as f:
		pickle.dump(feature_cols, f)

features = model.stages[-1].selectedFeatures
print("Feature Indices: {0}".format(features))
print("Feature Names: {0}".format([indexed_cols[i].replace("_indexed" , "") for i in features]))
