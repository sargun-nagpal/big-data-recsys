#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, percentile_approx, max, countDistinct, avg, min, coalesce, collect_set
from pyspark.mllib.evaluation import RankingMetrics

def evaluateMAP(spark, userID, k):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Load the recommendations
    recommendations_df = spark.read.parquet(f'hdfs://nyu-dataproc-m/user/sd5251_nyu_edu/recsys/baseline/pop_interactions_count_temporal_full_train.parquet', schema='recording_mbid STRING, score FLOAT')
    # Load the val/test set
    val_df = spark.read.parquet(f'hdfs:///user/hk3820_nyu_edu/interactions_test.parquet', schema="recording_msid STRING, user_id STRING, timestamp STRING, recording_mbid STRING")
    
    # Collect the user true items
    val_df= val_df.groupby("user_id").agg(collect_set("recording_mbid").alias('true_items'))
    
    # Collect the recommendations in a single list
    recommendations_df = recommendations_df.agg(collect_set("recording_mbid").alias('predicted_items'))
    
    # Cross join to get the recommendations at a user level 
    reco_and_labels_df = val_df.crossJoin(recommendations_df)
    reco_and_labels_rdd = reco_and_labels_df.rdd.map(lambda row: (row['predicted_items'], row['true_items']))

    # Compute the ranking metrics
    metrics = RankingMetrics(reco_and_labels_rdd)
    return metrics.precisionAt(k), metrics.meanAveragePrecisionAt(k), metrics.ndcgAt(k)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    k = 100

    # Call our main routine
    precision, mean_average_precision, ndcg = evaluateMAP(spark, userID, k)
    print("@%d: Precision: %.5f, MAP: %.7f, NDCG: %.5f"%(k, precision, mean_average_precision, ndcg))
