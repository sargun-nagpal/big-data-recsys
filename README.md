# An Implicit Feedback Music Recommender System
This is my final project for the Big Data course (DS-GA 1004) at NYU.

In this project, we used a 11.6 GB real-world dataset of music listening behavior containing data on ~8000 users, over 23 million tracks and 179 million user-track interactions to build an implicit feedback music recommender system using Spark.

We preprocessed and partitioned our data, implemented a baseline and Latent Factor model, and evaluated the model recommendations. We compared the multi-node performance with a single machine implementation, and investigated ways to accelerate inference using approximate search. The ALS model achieved a MAP@100 of 0.064 and NDCG@100 of 0.15 on the test set.

Detailed findings and analysis: [Report](Report.pdf)