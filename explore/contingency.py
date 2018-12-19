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

#chi_test = ChiSquareTest.test(df_features.limit(10000), "features", "HasDetections")

selector = ChiSqSelector(numTopFeatures=25, labelCol="HasDetections", featuresCol="features", outputCol="selectedFeatures")
model = selector.fit(df_features)
model_df = model.transform(df_features)
model.selectedFeatures
print("******    ChiSquare Selected Features   ******")
[feature_cols[i] for i in model.selectedFeatures]


data = [(0, 2.0, True, "1", "foo"), (1, 3.0, False, "2", "bar"),(0, 2.0, True, "1", "foo"), (1, 3.0, False, "2", "bar")]
cols = ["label", "real", "bool", "stringNum", "string"]
feature_cols = ["real", "bool", "stringNum", "string"]
x = spark.createDataFrame(data, cols)
h = FeatureHasher(numFeatures=4, inputCols=feature_cols, outputCol="features", categoricalCols=feature_cols)
x_df = h.transform(x)
x_df.show(truncate=False)

s = ChiSqSelector(numTopFeatures=2, labelCol="label", featuresCol="features", outputCol="selectedFeatures")
m = s.fit(x_df)
m_df = m.transform(x_df)
m_df.show(truncate=False)
s_df = m_df.select(*(m_df.columns[column_index] for column_index in m.selectedFeatures))
s_df.show(truncate=False)
m.selectedFeatures

