from __future__ import print_function
"""

Total Association Rules (df_rules): 					107,782,403
Total Association Rules, Antecedent = HasDetections_1:	 23,973,186

df = spark.read.table("fp_growth")
"""
import pickle

from pyspark.ml.feature import FeatureHasher
from pyspark.ml.feature import ChiSqSelectorModel
from pyspark.ml.fpm import FPGrowth

import pyspark.sql.functions as F

fp_growth_path = "output/fp_growth"
fp_growth_model_path = "output/fp_growth_model"

chi_model_path = "../explore/output/chi_model"
feature_path = "../explore/output/features.pkl"

feature_cols = None
model = None

try:
	model = ChiSqSelectorModel.load(chi_model_path)

	with open(feature_path, "rb") as f:
		feature_cols = pickle.load(f)
except:
	print("WARN: Output path missing")

selected_cols = [feature_cols[i] for i in model.selectedFeatures]
print("Selected Features Count: {0}".format(len(selected_cols)))
print("Selected Features: {0}".format(selected_cols))

df = spark.read.load("../datasets/train.csv", format="csv", sep=",", inferSchema="true", header="true")
df = df.sample(fraction=0.10, seed=3)
#df.cache()

all_columns = df.columns
label_col = ["HasDetections"]
meta_cols = ["MachineIdentifier"]
exclude_cols = ["DefaultBrowsersIdentifier", "OrganizationIdentifier", "GeoNameIdentifier", "LocaleEnglishNameIdentifier", "Census_ProcessorCoreCount", "Census_ProcessorClass", "Census_PrimaryDiskTotalCapacity", "Census_PrimaryDiskTypeName", "Census_SystemVolumeTotalCapacity", "Census_HasOpticalDiskDrive", "Census_TotalPhysicalRAM", "Census_InternalPrimaryDiagonalDisplaySizeInInches", "Census_InternalPrimaryDisplayResolutionHorizontal", "Census_InternalBatteryType", "Census_InternalBatteryNumberOfCharges"]
growth_cols = list(set(selected_cols) - set(exclude_cols))
growth_cols = list(label_col + growth_cols)

print("Selected Features without Exclusions Count: {0}".format(len(growth_cols)))

for column_name in growth_cols:
	df = df.withColumn(column_name + "_Item", F.concat(F.lit(column_name + "_"), F.col(column_name)))
	
item_cols = [column_name for column_name in df.columns if column_name.endswith("_Item")]
ordered_cols = list(label_col + meta_cols + item_cols)
df = df.select(*ordered_cols)

df = df.withColumn("items", F.split(F.concat_ws(",", *item_cols), ","))

print("Performing Fitting")

fp = FPGrowth(minSupport=0.002, minConfidence=0.7)
fpm = fp.fit(df)

print("Saving Models")

fp.write().overwrite().save(fp_growth_path)
fpm.write().overwrite().save(fp_growth_model_path)

print("Creating Frequent Items:")
df_frequent = fpm.freqItemsets
#df_frequent.cache()
#df_frequent.show(truncate=False)
#df_frequent.where(F.array_contains(df_frequent.items, "HasDetections_1")).show(truncate=False)
#df_frequent.where(F.array_contains(df_frequent.items, "HasDetections_1")).orderBy(df_frequent.freq.desc()).show(100, truncate=False)

print("Createing Association Rules:")
df_rules = fpm.associationRules
#df_rules.cache()

df_rules.write.saveAsTable("fp_growth_features_15", format="parquet", mode="overwrite", path="output/tables/fp_growth/features_15")
df_rules.show(truncate=False)

print("Executing Prediction")

test_data = spark.createDataFrame([(["Census_OSVersion_10.0.17134.228", "Census_OSBranch_rs4_release", "AVProductStatesIdentifier_53447", 
									"Census_GenuineStateName_IS_GENUINE", "Census_IsAlwaysOnAlwaysConnectedCapable_0", "SMode_0", "Platform_windows10", 
									"AVProductsEnabled_1", "AVProductsEnabled_1", "Census_IsFlightsDisabled_0", "IsSxsPassiveMode_0", "Census_IsPortableOperatingSystem_0", 
									"Census_DeviceFamily_Windows.Desktop", "Census_ProcessorManufacturerIdentifier_5", "OsBuild_17134", "HasTpm_1", "IsProtected_1"], )], ["items"])
test_results = fpm.transform(test_data)

"""

df_rules.where(F.array_contains(df_rules.consequent, "HasDetections_1")).orderBy(df_rules.confidence.desc()).select("antecedent").show(50, truncate=False)
df_rules.where(F.array_contains(df_rules.consequent, "HasDetections_1")).orderBy(df_rules.confidence.desc()).select("confidence").show(50, truncate=False)

test_data = spark.createDataFrame([(["Census_OSVersion_10.0.17134.228", "Census_OSBranch_rs4_release", "AVProductStatesIdentifier_53447", 
									"Census_GenuineStateName_IS_GENUINE", "Census_IsAlwaysOnAlwaysConnectedCapable_0", "SMode_0", "Platform_windows10", 
									"AVProductsEnabled_1", "AVProductsEnabled_1", "Census_IsFlightsDisabled_0", "IsSxsPassiveMode_0", "Census_IsPortableOperatingSystem_0", 
									"Census_DeviceFamily_Windows.Desktop", "Census_ProcessorManufacturerIdentifier_5", "OsBuild_17134", "HasTpm_1", "IsProtected_1"], )], ["items"])
test_results = fpm.transform(test_data)
"""


