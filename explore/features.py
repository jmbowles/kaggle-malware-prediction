from __future__ import print_function
"""

https://www.microsoft.com/en-us/wdsi/definitions/antimalware-definition-release-notes?RequestVersion=1.225.5.0
<p class="c-paragraph-3 article-a x-hidden-focus" id="availabledate">Definition available date:&nbsp;Mar 23, 2017 12:23 AM UTC</p>

1.225.5.0		Jun 29, 2016 11:41 PM UTC
1.227.8.0		Aug 18, 2016 01:30 AM UTC
1.229.9.0		Sep 23, 2016 01:12 AM UTC
1.231.7.0		Oct 20, 2016 12:16 AM UTC
1.237.0.0		Feb 22, 2017 11:56 PM UTC
1.237.5.0		Feb 23, 2017 01:06 AM UTC
1.239.0.0		Mar 23, 2017 12:23 AM UTC

1.261.13.0		Jan 19, 2018 03:55 AM UTC
1.261.18.0		Jan 19, 2018 05:10 AM UTC
1.261.22.0		Jan 19, 2018 07:10 AM UTC
1.261.77.0		Jan 20, 2018 05:36 PM UTC
1.261.92.0		Jan 22, 2018 11:15 AM UTC
1.261.252.0		Jan 25, 2018 08:51 PM UTC
1.261.551.0		Jan 31, 2018 07:24 PM UTC
1.261.637.0		Feb 01, 2018 09:33 PM UTC
1.261.674.0		Feb 02, 2018 01:12 PM UTC
1.261.728.0		Feb 03, 2018 06:26 PM UTC
1.261.791.0		Feb 05, 2018 09:27 PM UTC
1.261.1721.0	Apr 06, 2018 03:52 PM UTC

"""
import pyspark.sql.functions as F
from pyspark.sql.window import Window

print("Loading and Caching Data")
df = spark.read.load("../datasets/train.csv", format="csv", sep=",", inferSchema="true", header="true")
df.cache()

continuous_cols = ["Census_OSBuildNumber", "AVProductStatesIdentifier", "AVProductsInstalled", "OsBuild", "OsSuite", "IeVerIdentifier", 
					"Census_ProcessorCoreCount", "Census_TotalPhysicalRAM", "Census_InternalBatteryNumberOfCharges", "Census_OSBuildRevision"]
fill_0_cols = ["OrganizationIdentifier", "RtpStateBitfield", "Wdft_RegionIdentifier", "Census_ProcessorManufacturerIdentifier", "Census_IsFlightsDisabled", "Census_IsFlightingInternal", 
				"HasTpm", "Census_OSUILocaleIdentifier", "Census_IsVirtualDevice", "SMode", "IsBeta", "Wdft_IsGamer", "Census_IsAlwaysOnAlwaysConnectedCapable", 
				"Census_IsPortableOperatingSystem", "Census_OEMModelIdentifier", "Census_OEMNameIdentifier", "IsSxsPassiveMode", "Census_ThresholdOptIn", "CountryIdentifier", 
				"Census_IsWIMBootEnabled", "Census_OSInstallLanguageIdentifier", "Census_IsPenCapable", "Census_ProcessorModelIdentifier", "Census_IsSecureBootEnabled", "AutoSampleOptIn",
				"UacLuaenable", "Census_IsTouchEnabled", "Census_HasOpticalDiskDrive", "Firewall", "IsProtected", "Census_InternalPrimaryDiagonalDisplaySizeInInches", "AVProductsEnabled", "GeoNameIdentifier"]
fill_unknown_cols = ["Census_MDC2FormFactor", "OsBuildLab", "Platform", "Census_OSArchitecture", "Census_DeviceFamily", "Census_PowerPlatformRoleName", "Census_OSSkuName", 
						"SkuEdition", "Census_ActivationChannel", "OsPlatformSubRelease", "Census_ChassisTypeName", "Census_FlightRing", "Census_OSInstallTypeName", 
						"Processor", "Census_OSWUAutoUpdateOptionsName", "ProductName", "Census_OSEdition", "Census_GenuineStateName", "Census_PrimaryDiskTypeName", "PuaMode"]
 
df = df.replace("requireAdmin", "RequireAdmin", ["SmartScreen"])
df = df.replace("on", "1", ["SmartScreen"])
df = df.replace("On", "1", ["SmartScreen"])
df = df.replace("Enabled", "1", ["SmartScreen"])
df = df.replace("prompt", "Prompt", ["SmartScreen"])
df = df.replace("Promt", "Prompt", ["SmartScreen"])
df = df.replace("00000000", "0", ["SmartScreen"])
df = df.replace("off", "0", ["SmartScreen"])
df = df.replace("OFF", "0", ["SmartScreen"])
df = df.replace("warn", "Warn", ["SmartScreen"])
df = df.fillna("0", ["SmartScreen"])
df = df.fillna(0, continuous_cols + fill_0_cols)
df = df.fillna("UNKNOWN", fill_unknown_cols)
df = df.fillna(9999999, ["Census_PrimaryDiskTotalCapacity"])
df = df.fillna(1000000, ["Census_InternalPrimaryDisplayResolutionHorizontal", "Census_InternalPrimaryDisplayResolutionVertical"])
df = df.withColumn("Derived_AvSigVersion", F.regexp_replace(df.AvSigVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_AppVersion", F.regexp_replace(df.AppVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_EngineVersion", F.regexp_replace(df.EngineVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_OsVer", F.regexp_replace(df.OsVer, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_CensusOSVersion", F.regexp_replace(df.Census_OSVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_Firmware", df.Census_FirmwareManufacturerIdentifier + df.Census_FirmwareVersionIdentifier)
df = df.withColumn("Derived_VolumeCapacity", F.round(df.Census_SystemVolumeTotalCapacity / df.Census_PrimaryDiskTotalCapacity, 2))
df = df.withColumn("Derived_Resolution", df.Census_InternalPrimaryDisplayResolutionHorizontal * df.Census_InternalPrimaryDisplayResolutionVertical)

derived_cols = [col for col in df.columns if col.startswith("Derived_")]

df = df.fillna(0, derived_cols)

df.where(F.expr("Derived_AvSigVersion <> 0")) \
		.orderBy("HasDetections", "Derived_AvSigVersion") \
		.select("HasDetections", "Wdft_RegionIdentifier", "AvSigVersion", "Derived_AvSigVersion", "CountryIdentifier") \
		.groupBy("HasDetections", "Wdft_RegionIdentifier") \
		.agg(F.approx_count_distinct("CountryIdentifier"), F.min("Derived_AvSigVersion").alias("min_version"), F.max("Derived_AvSigVersion").alias("max_version")) \
		.orderBy("Wdft_RegionIdentifier", "HasDetections").show(50)

df.select("AvSigVersion").where(F.expr("Derived_AvSigVersion <> 0 and AvSigVersion like '1.261.%'")).distinct().orderBy("Derived_AvSigVersion").show(100)

window = Window.partitionBy("Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier")
detections = Window.partitionBy("HasDetections", "Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier")
total_detections = F.count("HasDetections").over(window)
detection_0 = F.count("HasDetections").over(detections)

df1 = df.where(F.expr("Derived_AvSigVersion <> 0")) \
		.select("HasDetections", "Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier") \
		.withColumn("Total_Detections", total_detections) \
		.withColumn("Count_0", F.when(F.expr("HasDetections == 0"), detection_0).otherwise(total_detections - detection_0)) \
		.withColumn("Count_1", total_detections - F.col("Count_0")) \
		.orderBy("Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier")
df1.select("HasDetections", "Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier", "Total_Detections", "Count_0", "Count_1").show(100)
df1.select("Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier", "Total_Detections", "Count_0", "Count_1").distinct().show(100)

'''
df1 = df.where(F.expr("Derived_AvSigVersion <> 0 and Wdft_RegionIdentifier = 15 and AvSigVersion = '1.237.0.0' and CountryIdentifier = 96")) \
		.groupBy("HasDetections", "Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier") \
		.agg(F.when(df.HasDetections == 0, F.count("HasDetections")).otherwise(0).alias("Count_0"), \
			 F.when(df.HasDetections == 1, F.count("HasDetections")).otherwise(0).alias("Count_1")) \
		.orderBy("Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier")
df1.show(100)
'''

df.select("MachineIdentifier", "HasDetections", "Wdft_RegionIdentifier", "AvSigVersion", "CountryIdentifier") \
	.where(F.expr("Wdft_RegionIdentifier = 15 and AvSigVersion = '1.237.0.0' and CountryIdentifier = 96")).show()




