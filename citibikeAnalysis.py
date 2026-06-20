from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.functions import col, unix_timestamp, count, min as spark_min, max as spark_max, avg
import os
from pyspark.sql.functions import count, avg, round, expr, col, when
from pyspark.sql.functions import (
    hour, dayofweek, when, col, lit, 
    radians, sin, cos, sqrt, atan2, round as spark_round
)
from pyspark.sql.functions import *

#---------------------------------------------PRVA TACKA----------------------------------------------------

spark = SparkSession.builder \
    .appName("CitiBike") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()


spark.sparkContext.setLogLevel("WARN")

#  Definicija šeme 
citibike_schema = StructType([
    StructField("ride_id", StringType(), True),
    StructField("rideable_type", StringType(), True),
    StructField("started_at", TimestampType(), True),
    StructField("ended_at", TimestampType(), True),
    StructField("start_station_name", StringType(), True),
    StructField("start_station_id", StringType(), True),
    StructField("end_station_name", StringType(), True),
    StructField("end_station_id", StringType(), True),
    StructField("start_lat", DoubleType(), True),
    StructField("start_lng", DoubleType(), True),
    StructField("end_lat", DoubleType(), True),
    StructField("end_lng", DoubleType(), True),
    StructField("member_casual", StringType(), True)
])

#  Učitavanje podataka
data_folder = "temp" 
csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]

df = spark.read \
          .option("header", "true") \
          .schema(citibike_schema) \
          .csv(csv_files)

#  Validacija učitavanja
print("\n--- Validacija učitavanja ---")
df.printSchema()
initial_count = df.count()
print(f"Ukupno učitanih zapisa: {initial_count:,}")

if initial_count == 0:
    print("GRESKA: Nema podataka!")
    spark.stop()
    exit(1)

# Čišćenje podatak
df = df.withColumn(
    "trip_duration_seconds",
    unix_timestamp(col("ended_at")) - unix_timestamp(col("started_at"))
)

print("\n--- Primjena filtera za čišćenje ---")

# Uslovi:
# 1. Nema NULL vrijednosti u ključnim kolonama
# 2. Trajanje vožnje je logično (npr. > 60 sekundi i < 24 sata)
# 3. Koordinate su unutar NYC opsega (opcionalno, ali dobra praksa)

df_clean = df.filter(
    (col("ride_id").isNotNull()) &
    (col("started_at").isNotNull()) &
    (col("ended_at").isNotNull()) &
    (col("start_station_id").isNotNull()) &
    (col("end_station_id").isNotNull()) &
    (col("trip_duration_seconds") > 60) &       # Min 1 minuta
    (col("trip_duration_seconds") < 86400) &    # Max 24 sata
    (col("start_lat").between(40.4, 41.0)) &
    (col("start_lng").between(-74.5, -73.5))
)


# Validacija nakon čišćenja
cleaned_count = df_clean.count()
removed_count = initial_count - cleaned_count
percent_removed = (removed_count / initial_count * 100) if initial_count > 0 else 0

print(f"Originalni broj: {initial_count:,}")
print(f"Očišćeni broj:   {cleaned_count:,}")
print(f"Uklonjeno:       {removed_count:,} ({percent_removed:.2f}%)")

# Čuvanje rezultata (Stavka zahtijeva čuvanje uzorka ako je set velik)
output_path = "citibike_results/cleaned_sample"
print(f"\nČuvanje uzorka (prvih 100 redova) u: {output_path}")

df_clean.drop("trip_duration_seconds") \
        .limit(100) \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv(output_path)


#------------------------------------ DRUGA TACKA------------------------------------

# Kreiranje vremenskih karakteristika i trajanja
df_features = df_clean.withColumn("duration_seconds", col("ended_at").cast("long") - col("started_at").cast("long"))\
                    .withColumn("duration_minutes", spark_round(col("duration_seconds") / 60, 2))\
                    .withColumn("start_hour", hour(col("started_at")))\
                    .withColumn("day_of_week", dayofweek(col("started_at")))\
                    .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), True).otherwise(False))

# Round Trip indicator
df_features = df_features.withColumn("is_round_trip", col("start_station_id") == col("end_station_id"))

# Dodatana inzenjerska karakteristika: Haversine formula za udaljenost
lat1 = radians(col("start_lat"))
lon1 = radians(col("start_lng"))
lat2 = radians(col("end_lat"))
lon2 = radians(col("end_lng"))

# Haversine formula
dlon = lon2 - lon1
dlat = lat2 - lat1
a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * asin(sqrt(a))
R = 6371.0  # Radius Zemlje u kilometrima

df_features = df_features.withColumn("distance_km", spark_round(R * c, 3))

output_path_features = "citibike_results/cleaned_with_features"
print(f"\nČuvanje rezultata sa dodatim karakteristikama u: {output_path_features}")
df_features.limit(100) \
           .write.mode("overwrite") \
           .option("header", "true") \
           .csv(output_path_features)

#------------------------------------ TRECA TACKA------------------------------------

# Agregacija podataka
df_hour_analysis = df_features.groupBy("is_weekend", "start_hour", "member_casual")\
    .agg(
        count("ride_id").alias("total_rides"),
        round(avg("duration_minutes"), 2).alias("avg_duration_minutes"),
        expr("round(percentile_approx(duration_minutes, 0.5), 2)").alias("median_duration_min")
    )

final_hourly_stats = df_hour_analysis \
    .withColumn("day_type", when(col("is_weekend") == True, "Vikend").otherwise("Radni dan")) \
    .select(
        "day_type", 
        "member_casual", 
        "start_hour", 
        "total_rides",          
        "avg_duration_minutes", 
        "median_duration_min"
    ) \
    .orderBy("is_weekend", "member_casual", "start_hour")

print("Rezultati analize po satu (prikaz prvih 50 redova):")
final_hourly_stats.show(50, truncate=False)

output_path_hourly = "citibike_results/hourly_usage_analysis"
print(f"Čuvanje rezultata analize u: {output_path_hourly}")

final_hourly_stats.coalesce(1) \
                  .write.mode("overwrite") \
                  .option("header", "true") \
                  .csv(output_path_hourly)

#------------------------------------ CETVRTA TACKA ------------------------------------

# Analiza top startnih lokacija i dodatne karakteristike
MIN_RIDES = 500

df_station_profile = df_features.filter(col("start_station_id").isNotNull())\
    .groupBy("start_station_name")\
    .agg(
        # Metrika 1 - Ukupan promet
        count("ride_id").alias("total_rides"),
        # Metrika 2 - Udio clanova
        round(avg(when(col("member_casual") == "member", 1).otherwise(0)), 2).alias("member_ratio"),       
        # Metrika 3 - Udio round-trip voznji
        round(avg(when(col("is_round_trip") == True, 1).otherwise(0)), 2).alias("round_trip_ratio")       
    )

# Filtriranje stanica sa najmanje MIN_RIDES voznji
df_station_profile_filtered = df_station_profile.filter(col("total_rides") >= MIN_RIDES)

# Sortiranje po ukupnom prometu i uzimanje top 10
top_round_trip_stations = df_station_profile_filtered.orderBy(col("round_trip_ratio").desc()).limit(10)
top_busy_stations = df_station_profile_filtered.orderBy(col("total_rides").desc()).limit(10)

print("Top 10 stanica po udjelu round-trip vožnji:")
top_round_trip_stations.show(truncate=False)

print("Top 10 najprometnijih stanica:")
top_busy_stations.show(truncate=False)

# Čuvanje rezultata analize stanica
output_path_stations = "citibike_results/station_profile_analysis"
print(f"Čuvanje rezultata analize stanica u: {output_path_stations}")
df_station_profile_filtered.coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(output_path_stations)

#-------------------------------------------TACKA PETA------------------------------------------

# Analiza najcescih relacija
MIN_ROUTE_RIDES = 100

df_routes = df_features.filter(
        (col("start_station_name").isNotNull()) & 
        (col("end_station_name").isNotNull())
    ) \
    .groupBy("start_station_name", "end_station_name") \
    .agg(
        count("ride_id").alias("total_rides"),
        expr("round(percentile_approx(duration_minutes, 0.5), 2)").alias("median_duration_min"),
        round(stddev("duration_minutes"), 2).alias("stddev_duration_min")
    )

df_top_routes = df_routes.filter(col("total_rides") >= MIN_ROUTE_RIDES) \
    .withColumn("route_name", concat(col("start_station_name"), lit(" -> "), col("end_station_name"))) \
    .select("route_name", "total_rides", "median_duration_min", "stddev_duration_min")

df_result_routes = df_top_routes.orderBy(col("total_rides").desc())
print("Top relacije (prvih 10):")
df_result_routes.show(10, truncate=False)

# Čuvanje rezultata analize relacija
output_path_routes = "citibike_results/route_analysis"
print(f"Čuvanje rezultata analize relacija u: {output_path_routes}")
df_result_routes.limit(100) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(output_path_routes)

#-------------------------------------------SESTA TACKA------------------------------------

# Analiza round trip voznji
df_rt_analysis = df_features.groupBy("member_casual", "rideable_type") \
    .agg(
        # Ukupan broj vožnji u toj grupi
        count("ride_id").alias("total_rides"),
        
        # Broj samo round-trip vožnji
        sum(when(col("is_round_trip") == True, 1).otherwise(0)).alias("rt_count"),
        
        # Tipično trajanje SAMO za round-trip vožnje (koristimo CASE unutar percentile funkcije)
        expr("round(percentile_approx(CASE WHEN is_round_trip = true THEN duration_minutes END, 0.5), 2)") \
            .alias("rt_median_duration_min")
    )

df_rt_final = df_rt_analysis.withColumn(
    "rt_percentage", 
    round((col("rt_count") / col("total_rides") * 100), 2)
)

df_rt_display = df_rt_final.select(
    "member_casual", 
    "rideable_type", 
    "total_rides", 
    "rt_count", 
    "rt_percentage", 
    "rt_median_duration_min"
).orderBy("member_casual", "rideable_type")

print("Analiza round-trip vožnji:")
df_rt_display.show(5, truncate=False)

# Čuvanje rezultata analize round-trip vožnji
output_path_rt = "citibike_results/round_trip_analysis"
print(f"Čuvanje rezultata analize round-trip vožnji u: {output_path_rt}")
df_rt_display.coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(output_path_rt)

#-------------------------------SEDMA TACKA--------------------------------------------

weather_csv_path = "KJRB0.csv"

# Ucitavanje vremenskih podataka
df_weather = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(weather_csv_path)  

# Ciscenje i flitriranje
df_weather_processed = df_weather.withColumn(
        "ts_str", 
        concat(
            col("year"), lit("-"), 
            col("month"), lit("-"), 
            col("day"), lit(" "), 
            col("hour"), lit(":00:00")
        )
    ).withColumn(
        "weather_ts_utc", to_timestamp(col("ts_str")) # Ovo je vrijeme u UTC
    ).withColumn(
        "weather_ts", from_utc_timestamp(col("weather_ts_utc"), "America/New_York") # Konverzija u NY vrijeme
    )

df_weather_filtered = df_weather_processed.filter(
        (col("year") == 2024) & (col("month") == 10)
    )

# Krerianje kljuca za join (Bucket) - svodjenje na sat
df_weather_filtered = df_weather_filtered.withColumn(
        "ts_join_key", date_trunc("hour", col("weather_ts"))
    )

# Kategorija temperature
df_features_weather = df_weather_filtered.withColumn(
        "temp_category",
        when(col("temp") < 5, "Vrlo hladno (<5°C)")
        .when((col("temp") >= 5) & (col("temp") < 10), "Hladno (5-10°C)")
        .when((col("temp") >= 10) & (col("temp") < 15), "Umjereno (10-15°C)")
        .when((col("temp") >= 15) & (col("temp") < 20), "Toplo (15-20°C)")
        .otherwise(" Vrlo toplo (>=20°C)")
    )
# Indikator padavina
df_features_weather = df_features_weather.withColumn(
    "rain_condition",
    when((col("prcp").isNull()) | (col("prcp") == 0), "Dry")
    .otherwise("Rainy")
)

# Kategorija vjetra
df_features_weather = df_features_weather.withColumn(
    "wind_condition",
    when(col("wspd") > 20, "Windy").otherwise("Calm")
)

df_weather_final = df_features_weather.select(
        "ts_join_key", 
        "temp", 
        "temp_category", 
        "prcp", 
        "rain_condition", 
        "wspd",
        "wind_condition"
    )

print("Prikaz pripremljenih vremenskih podataka:")
df_weather_final.show(5, truncate=False)

# Čuvanje pripremljenog seta
output_path_weather = "citibike_results/weather_prepared"
df_weather_final.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path_weather)

#----------------------------- OSMA TACKA-----------------------------------

# Spajanje CitiBike i Meteostat hourly podatke
df_rides_for_join = df_features.withColumn("ts_join_key", date_trunc("hour", col("started_at")))
df_joined = df_rides_for_join.join(broadcast(df_weather_final), on="ts_join_key", how="inner")

# Poredimo rainy vs dry 
df_rain_impact = df_joined.groupBy(
        "member_casual", 
        "is_weekend", 
        "rain_condition"
    ).agg(
        count("ride_id").alias("total_rides"),
        spark_round(avg("duration_minutes"), 2).alias("avg_duration_min")
    )

final_rain_impact = df_rain_impact.withColumn(
        "day_type", 
        when(col("is_weekend") == True, "Vikend").otherwise("Radni dan")
    ).select(
        "day_type",
        "member_casual",
        "rain_condition",
        "total_rides",
        "avg_duration_min"
    ).orderBy("day_type", "member_casual", "rain_condition")

print("Rezultati analize uticaja kiše:")
final_rain_impact.show(10, truncate=False)

output_path_rain = "citibike_results/rain_impact_analysis"
print(f"Čuvanje rezultata u: {output_path_rain}")

final_rain_impact.coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(output_path_rain)

#-----------------------------DEVETA TACKA------------------------------------

df_temp_analysis = df_joined.groupBy("temp_category", "member_casual")\
                            .agg(
                                count("ride_id").alias("total_rides"),
                                spark_round(avg("duration_minutes"), 2).alias("avg_duration_min")   
                            )
final_temp_stats = df_temp_analysis.orderBy("temp_category", "member_casual")

print("Rezultati analize temperature:")
final_temp_stats.show(10, truncate=False)

output_path_temp = "citibike_results/temperature_analysis"
final_temp_stats.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path_temp)