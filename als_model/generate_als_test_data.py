#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Read in test data
    track_interactions_test = spark.read.parquet(f'hdfs:/user/hk3820_nyu_edu/interactions_test.parquet', schema="recording_msid STRING, user_id INT, timestamp STRING, recording_mbid STRING").select("user_id", "recording_mbid")

    # Read in the track_id - mbid mapping
    track_id_mbid_mapping = spark.read.parquet('hdfs:/user/sd5251_nyu_edu/recsys/als/track_id_mbid_mapping.parquet', schema="track_id INT, recording_mbid STRING")

    # Join to get the mapping on test
    track_interactions_test = track_interactions_test.join(track_id_mbid_mapping, "recording_mbid", "inner").select("user_id", "track_id")

    als_test = track_interactions_test.groupby("track_id", "user_id") \
                                            .agg((count("*")).alias("interactions_count")) 
    als_test.show(10)
    # Save train data
    als_test.write.parquet(f"hdfs:/user/{userID}/recsys/als/als_test_data.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
