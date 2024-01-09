#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, udf, collect_set
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.types import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import StorageLevel
import time

def evaluate(reco_df, val_df):
    val_df = val_df.groupby("user_id").agg(collect_set("track_id").alias('true_items'))
    #val_df.show(10)
    #reco_df.show(10)
    reco_and_labels_df = reco_df.join(val_df, "user_id", "inner")
    reco_and_labels_df.persist(StorageLevel.MEMORY_AND_DISK)
    reco_and_labels_df.show(10)
    reco_and_labels_rdd = reco_and_labels_df.rdd.map(lambda row: (row['reco'], row['true_items']))
    metrics = RankingMetrics(reco_and_labels_rdd)
    k=100
    return metrics.precisionAt(k), metrics.meanAveragePrecisionAt(k), metrics.ndcgAt(k)

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Read in the Train and Validation Data
    als_val_data = spark.read.parquet(f'hdfs:/user/sd5251_nyu_edu/recsys/als/als_test_data.parquet', schema="track_id INT, user_id INT, interactions_count INT")
    
    # Read in the recommendations generated from the model
    reco = spark.read.parquet("hdfs:/user/sd5251_nyu_edu/recsys/als/13_05_23_02_43_37_alpha_0_5_rank_25_reg_0_1_als_50_reco.parquet", schema="user_id INT, recommendations ARRAY<INT>")
    # Evaluate Recommendations on Validation Set
    evaluation_start_time = time.time()
    precision, map_metric, ndcg = evaluate(reco, als_val_data)
    evaluation_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - evaluation_start_time))
    print(f"Recommendation Complete. Evaluation Time Taken: {evaluation_elapsed}")
    print("-"*50)
    print("Precision: %.5f, MAP: %.7f, NDCG: %.5f"%(precision, map_metric, ndcg))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
