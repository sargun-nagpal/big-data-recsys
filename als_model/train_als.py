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
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import StorageLevel
import time

def train_als(als_data, maxIter = 10, alpha=0.1, regParam = 0.01, rank=10):
    USER_ID = "user_id"
    TRACK_ID = "track_id"
    RATING_PROXY = "interactions_count"
    als = ALS(maxIter = maxIter, \
                rank=rank, \
                regParam = regParam, \
                implicitPrefs = True, \
                alpha = alpha, \
                userCol = USER_ID, \
                itemCol = TRACK_ID, \
                ratingCol = RATING_PROXY, \
                coldStartStrategy="drop")
    start = time.time()
    als_model = als.fit(als_data)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"Training Complete. Time Taken: {elapsed}")
    #print("Time Taken to train:", time.time()-start)
    return als_model

def evaluate(reco_df, val_df):
    val_df = val_df.groupby("user_id").agg(collect_set("track_id").alias('true_items'))
    #val_df.show(10)
    #reco_df.show(10)
    reco_and_labels_df = reco_df.join(val_df, "user_id", "left")
    reco_and_labels_df.persist(StorageLevel.MEMORY_AND_DISK)
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
    als_train_data = spark.read.parquet(f'hdfs:/user/sd5251_nyu_edu/recsys/als/als_train_data.parquet', schema="track_id INT, user_id INT, interactions_count INT")
    als_val_data = spark.read.parquet(f'hdfs:/user/sd5251_nyu_edu/recsys/als/als_val_data.parquet', schema="track_id INT, user_id INT, interactions_count INT")
    
    # Hyperparamter Spcecification
    RANK = 25
    ALPHA = 0.5
    REGPARAM = 0.1
    
    # Train ALS
    als_model = train_als(als_data=als_train_data, maxIter=10, rank=RANK, alpha=ALPHA, regParam=REGPARAM)
    MODEL_ID = time.strftime("%d_%m_%y_%H_%M_%S", time.gmtime(time.time()))+"_alpha_"+str(ALPHA).replace(".", "_")+"_rank_"+str(RANK)+"_reg_"+str(REGPARAM).replace(".", "_")+"_als"
    als_model.save("hdfs:/user/sd5251_nyu_edu/recsys/als/models/"+MODEL_ID)
    # Recommendation Phase
    NUM_RECOMMENDATIONS = 100
    #user_subset = als_train_data.limit(1000).select('user_id')
    start = time.time()
    #reco = als_model.recommendForUserSubset(user_subset, 20)
    reco = als_model.recommendForAllUsers(NUM_RECOMMENDATIONS)
    def extract_track_id(row_list):
        return [row[0] for row in row_list]

    get_rec_only = udf(extract_track_id, ArrayType(IntegerType()))
    reco = reco.withColumn('reco', get_rec_only('recommendations')).select('user_id', 'reco')
    
    # Evaluate Recommendations on Validation Set
    evaluation_start_time = time.time()
    precision, map_metric, ndcg = evaluate(reco, als_val_data)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    evaluation_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - evaluation_start_time))
    print(f"Recommendation Complete. Time Taken: {elapsed}, Evaluation Time Taken: {evaluation_elapsed}")
    print("-"*50)
    print("rank: %d, alpha: %.2E, regParam: %.2E \nPrecision: %.5f, MAP: %.7f, NDCG: %.5f"%(RANK, ALPHA, REGPARAM, precision, map_metric, ndcg))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
