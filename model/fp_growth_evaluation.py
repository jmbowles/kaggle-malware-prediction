from __future__ import print_function
"""


"""



from pyspark.ml.fpm import FPGrowth, FPGrowthModel

import pyspark.sql.functions as F

fp_growth_path = "output/fp_growth"
fp_growth_model_path = "output/fp_growth_model"


print("Loading Models")

fp = FPGrowth.load(fp_growth_path)
fpm = FPGrowthModel.load(fp_growth_model_path)
#fpm = fp.fit(df)

#df_rules = spark.read.table("fp_growth_features_15")

print("Executing Prediction")

test_data = spark.createDataFrame([(["Census_OSVersion_10.0.17134.228", "Census_OSBranch_rs4_release", "AVProductStatesIdentifier_53447", 
									"Census_GenuineStateName_IS_GENUINE", "Census_IsAlwaysOnAlwaysConnectedCapable_0", "SMode_0", "Platform_windows10", 
									"AVProductsEnabled_1", "AVProductsEnabled_1", "Census_IsFlightsDisabled_0", "IsSxsPassiveMode_0", "Census_IsPortableOperatingSystem_0", 
									"Census_DeviceFamily_Windows.Desktop", "Census_ProcessorManufacturerIdentifier_5", "OsBuild_17134", "HasTpm_1", "IsProtected_1"], )], ["items"])
test_results = fpm.transform(test_data)

"""

df_rules.where(F.array_contains(df_rules.antecedent, "HasDetections_1")).orderBy(df_rules.confidence.desc()).show(50, truncate=False)
df_rules.where(F.array_contains(df_rules.consequent, "HasDetections_1")).orderBy(df_rules.confidence.desc()).show(50, truncate=False)


"""


