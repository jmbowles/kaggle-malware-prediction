from __future__ import print_function

from pyspark.ml.feature import FeatureHasher
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

from pyspark.sql import Column
import pyspark.sql.functions as F


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

