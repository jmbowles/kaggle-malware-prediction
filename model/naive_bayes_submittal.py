from __future__ import print_function
"""


"""
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

import pyspark.sql.functions as F

pipeline_model_path = "output/naive_bayes_pipeline_model"
csv_path = "output/submit/submittal_5.csv"

test = spark.read.load("../datasets/test.csv", format="csv", sep=",", inferSchema="true", header="true")
test = test.withColumn("Derived_Firmware", F.concat(test.Census_FirmwareManufacturerIdentifier, F.lit("_"), test.Census_FirmwareVersionIdentifier))
test = test.withColumn("Derived_Processor", F.concat(test.Census_ProcessorManufacturerIdentifier, F.lit("_"), test.Census_ProcessorModelIdentifier))
test = test.withColumn("Derived_Installer", F.concat(test.Census_OSInstallTypeName, F.lit("_"), test.Census_OSInstallLanguageIdentifier))

print("Loading Pipeline")
pipeline_model = PipelineModel.load(pipeline_model_path)

print("Fitting for Submittal")
predictions = pipeline_model.transform(test)
predictions.select("MachineIdentifier", "probability", "prediction").show(truncate=False)

print("Creating CSV for Submittal")

# Silly workaround for extracting an element from a dense or sparse vector. Probability column is a vector, with probs for each label
# https://stackoverflow.com/questions/39555864/how-to-access-element-of-a-vectorudt-column-in-a-spark-dataframe
def vector_item_(vector_column, index):
    try:
        return float(vector_column[index])
    except ValueError:
        return None

vector_item = F.udf(vector_item_, DoubleType())

df_submit = predictions.withColumn("Label_0", vector_item("probability", F.lit(0)))
df_submit = df_submit.withColumn("Label_1", vector_item("probability", F.lit(1)))
df_submit = df_submit.withColumn("HasDetections", df_submit.Label_1)
df_submit = df_submit.select("MachineIdentifier", "HasDetections")

# Yet another workaround to write to a CSV file
df_submit.coalesce(1).toPandas().to_csv(csv_path, header=True, index=False)

print("Total rows written to file: {0}".format(df_submit.count()))







