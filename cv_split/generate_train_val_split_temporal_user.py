import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number


def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    tracks = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet', schema="recording_msid STRING, artist_name STRING, track_name STRING, recording_mbid STRING")
    interactions = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet', schema="user_id STRING, recording_msid STRING, timestamp STRING")

    interactions_tracks = interactions.join(tracks, tracks.recording_msid == interactions.recording_msid).select(interactions.user_id, 
                                                                                                            interactions.recording_msid,
                                                                                                            F.to_date(interactions.timestamp).alias('timestamp'),
                                                                                                        tracks.recording_mbid)

    interactions_tracks = interactions_tracks.withColumn("recording_mbid",F.coalesce(interactions_tracks.recording_mbid, interactions_tracks.recording_msid)) 
    window = Window.partitionBy(interactions_tracks["user_id"]).orderBy(interactions_tracks["user_id"])
    interactions_tracks_count = interactions_tracks.groupBy("user_id").count().withColumnRenamed("user_id", "user_id_grp")
    interactions_tracks = interactions_tracks.join(interactions_tracks_count, col("user_id") == col("user_id_grp"), "left").drop("user_id_grp")
    interactions_tracks = interactions_tracks.orderBy('timestamp')
    interactions_tracks = interactions_tracks.select("user_id", "recording_msid", "timestamp", "recording_mbid", "count", row_number().over(window).alias("row_number"))
    interactions_tracks.write.parquet('hdfs:/user/hk3820_nyu_edu/interactions_tracks_temporal_user.parquet')
    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)


