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
from pyspark.ml.recommendation import ALSModel

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    MODEL_ID = "09_05_23_19_52_22_alpha_0_5_rank_20_reg_0_01_als"
    MODEL_PATH = "hdfs:/user/sd5251_nyu_edu/recsys/als/models/" + MODEL_ID
    als_model = ALSModel.load(MODEL_PATH)
    
    item_factors = als_model.itemFactors
    user_factors = als_model.userFactors

    FACTOR_PATH = "hdfs:/user/sd5251_nyu_edu/recsys/als/factors/" + MODEL_ID
    item_factors.write.parquet(FACTOR_PATH+"_if.parquet")
    user_factors.write.parquet(FACTOR_PATH+"_uf.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
