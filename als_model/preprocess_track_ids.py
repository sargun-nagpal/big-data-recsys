#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, monotonically_increasing_id
from pyspark.ml.feature import StringIndexer

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    interactions_tracks = spark.read.parquet(f'hdfs:/user/hk3820_nyu_edu/interactions_tracks_temporal_user.parquet',
                                             schema='user_id INT, recording_msid STRING, timestamp STRING, recording_mbid STRING, count INT, row_number INT')
    # track_indexer = StringIndexer(inputCol="recording_mbid", outputCol="track_id")
    # interactions_tracks = track_indexer.fit(interactions_tracks).transform(interactions_tracks)
    tracks_df = interactions_tracks.select("recording_mbid").distinct()
                                            #.agg((count("*")).alias("count"))
    # tracks_df = tracks_df.withColumn("track_id", monotonically_increasing_id())
    tracks_df.createOrReplaceTempView('tracks_df')
    tracks_df = spark.sql('select row_number() over (order by recording_mbid) as track_id, recording_mbid from tracks_df')
    it_df = interactions_tracks.join(tracks_df, ["recording_mbid"], 'left').select("user_id", "track_id", "row_number", "count")
    
    #Train Data
    split_percentage = 80
    interactions_train = it_df.filter(col("row_number") <= col("count") * split_percentage / 100).select("user_id", "track_id")

    interactions_train.show(10)
    # Save train data
    interactions_train.write.parquet(f"hdfs:/user/{userID}/recsys/als/interactions_str_ind_trackid_train.parquet")

    interactions_val = it_df.filter(col("row_number") > col("count") * split_percentage / 100).select("user_id", "track_id")
    interactions_val.show(10)
    # Save val data
    interactions_val.write.parquet(f"hdfs:/user/{userID}/recsys/als/interactions_str_ind_trackid_val.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
