import os

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

    interactions_tracks = spark.read.parquet(f'hdfs:/user/hk3820_nyu_edu/interactions_tracks_temporal_user.parquet',
                                             schema='user_id INT, recording_msid STRING, timestamp STRING, recording_mbid STRING, count INT, row_number INT')
    split_percentage = 80
    df_train = interactions_tracks.filter(col("row_number") <= col("count") * split_percentage / 100)
    df_train.write.parquet("hdfs:/user/hk3820_nyu_edu/interactions_train_temporal_user.parquet")

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
