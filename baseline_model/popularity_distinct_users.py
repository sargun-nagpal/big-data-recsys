#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, desc, first, count, coalesce

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Calculate Popularity based on #distinct-users for each item
    track_interactions_train = spark.read.parquet(f'hdfs:///user/hk3820_nyu_edu/interactions_tracks_temporal_user.parquet', schema="recording_msid STRING, user_id STRING, timestamp STRING, recording_mbid STRING, row_number INT")

    results = track_interactions_train.select("recording_mbid", "user_id") \
                                            .groupby("recording_mbid") \
                                            .agg(
                                                (countDistinct("user_id")).alias("score")
                                                ) \
                                            .sort(desc("score")) \
                                            .limit(100)
    results.show(10)
    # Save results
    results.write.parquet(f"hdfs:/user/{userID}/recsys/baseline/pop_interactions_distinct_user_full_train.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
