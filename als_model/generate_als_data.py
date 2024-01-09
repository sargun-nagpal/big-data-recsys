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

    # Read in train data
    track_interactions_train = spark.read.parquet(f'hdfs:///user/sd5251_nyu_edu/recsys/als/interactions_str_ind_trackid_train.parquet', schema="user_id INT, track_id INT")
    als_train = track_interactions_train.groupby("track_id", "user_id") \
                                            .agg((count("*")).alias("interactions_count")) 
    als_train.show(10)
    # Save train data
    als_train.write.parquet(f"hdfs:/user/{userID}/recsys/als/als_train_data.parquet")

    #Read in Val Data
    track_interactions_val = spark.read.parquet(f'hdfs:///user/sd5251_nyu_edu/recsys/als/interactions_str_ind_trackid_val.parquet', schema="user_id INT, track_id INT")
    als_val = track_interactions_val.groupby("track_id", "user_id") \
                                            .agg((count("*")).alias("interactions_count")) 
    als_val.show(10)
    # Save val data
    als_val.write.parquet(f"hdfs:/user/{userID}/recsys/als/als_val_data.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
