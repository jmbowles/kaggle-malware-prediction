from __future__ import print_function

from pyspark.ml.feature import FeatureHasher
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

from pyspark.sql import Column
import pyspark.sql.functions as F


df = spark.read.load("../datasets/train.csv", format="csv", sep=",", inferSchema="true", header="true")
df.cache()

total_detections = df.select("HasDetections").where(df.HasDetections==1).count()

print("Total Rows: {0}".format(df.count()))
print("Total HasDetections: {0}".format(total_detections))
print("******    Crosstabulations   ******")
df.crosstab("HasDetections", "SkuEdition").show(truncate=False)
df.crosstab("HasDetections", "ProductName").show(truncate=False)
df.crosstab("HasDetections", "AVProductsEnabled").show(truncate=False)
df.crosstab("HasDetections", "IsBeta").show(truncate=False)
df.crosstab("HasDetections", "Platform").show(truncate=False)
df.crosstab("HasDetections", "Census_DeviceFamily").show(truncate=False)
df.crosstab("HasDetections", "Census_OSInstallTypeName").show(truncate=False)

all_columns = df.columns
label_col = ["HasDetections"]
meta_cols = ["MachineIdentifier"]
#feature_cols = ["SkuEdition", "ProductName", "AVProductsEnabled", "IsBeta", "Platform", "Census_DeviceFamily", "Census_OSInstallTypeName"]
feature_cols = list(set(all_columns) - set(label_col) - set(meta_cols))
ordered_cols = list(label_col + meta_cols + feature_cols)

#hasher = FeatureHasher(numFeatures=len(feature_cols), inputCols=feature_cols, outputCol="features",  categoricalCols=feature_cols)
hasher = FeatureHasher(numFeatures=len(feature_cols), inputCols=feature_cols, outputCol="features",  categoricalCols=feature_cols)
df_features = df.sample(fraction=0.50, seed=3)
df_features = df.select(*ordered_cols)
df_features = hasher.transform(df_features)

chi_test = ChiSquareTest.test(df_features, "features", "HasDetections")

