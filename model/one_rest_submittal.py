from __future__ import print_function
"""


"""
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

import pyspark.sql.functions as F

pipeline_model_path = "output/one_vs_rest_pipeline_model"
csv_path = "output/submit/submittal_6.csv"

print("Loading Data")
test = spark.read.load("../datasets/test.csv", format="csv", sep=",", inferSchema="true", header="true")

'''
test = test.withColumn("Derived_Firmware", F.concat(test.Census_FirmwareManufacturerIdentifier, F.lit("_"), test.Census_FirmwareVersionIdentifier))
test = test.withColumn("Derived_Processor", F.concat(test.Census_ProcessorManufacturerIdentifier, F.lit("_"), test.Census_ProcessorModelIdentifier))
test = test.withColumn("Derived_Installer", F.concat(test.Census_OSInstallTypeName, F.lit("_"), test.Census_OSInstallLanguageIdentifier))
'''
print("Loading Pipeline")
pipeline_model = PipelineModel.load(pipeline_model_path)

print("Fitting for Submittal")
predictions = pipeline_model.transform(test)
predictions.select("MachineIdentifier", "prediction").show(truncate=False)

print("Creating CSV for Submittal")
df_submit = predictions.withColumn("HasDetections", predictions.prediction.cast("integer"))
df_submit = df_submit.select("MachineIdentifier", "HasDetections")

# Yet another workaround to write to a CSV file
df_submit.coalesce(1).toPandas().to_csv(csv_path, header=True, index=False)

print("Total rows written to file: {0}".format(df_submit.count()))