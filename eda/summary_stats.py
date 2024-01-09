#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, percentile_approx, max, countDistinct, avg, min, coalesce

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Load the boats.txt and sailors.json data into DataFrame
    users_train = spark.read.parquet(f'/user/bm106_nyu_edu/1004-project-2023/users_train.parquet', schema='user_id STRING, user_name STRING')
    tracks_train = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet', schema="recording_msid STRING, artist_name STRING, track_name STRING, recording_mbid STRING")
    interactions_train = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet', schema="user_id STRING, recording_msid STRING, timestamp STRING")

    #User Descriptive Stats
    user_interactions = users_train.join(interactions_train, "user_id")

    user_interactions = user_interactions.groupBy("user_id").agg(count("recording_msid").alias("interaction_count"))
    perc_range = [i/100 for i in range(1, 6)]
    perc_range += [i/100 for i in range(10, 100, 5)]
    perc_range += [i/100 for i in range(95, 101)]
    args = []
    for perc in perc_range:
        label = str(perc*100) +"%"
        args.append(percentile_approx("interaction_count", perc).alias(label))
    user_interactions.agg(*args).show()
#    user_interactions.summary().show()

    #Track Descriptive Stats
    track_interactions = tracks_train.join(interactions_train, "recording_msid")
    track_interactions = track_interactions.withColumn("recording_mbid",coalesce(track_interactions.recording_mbid, track_interactions.recording_msid)) 

    track_interactions = track_interactions.groupBy("recording_mbid").agg(count("track_name").alias("interaction_count"))
    track_interactions.agg(*args).show()
#    track_interactions.summary().show()



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
