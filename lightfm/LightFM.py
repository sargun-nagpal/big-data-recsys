import pandas as pd
import pickle
import time
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from scipy import sparse
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import count, udf, collect_set
from pyspark.mllib.evaluation import RankingMetrics


def train_model(train_sparse_dataset, 
                item_alpha=0.0, 
                user_alpha=0.0, 
                no_components=10, 
                loss='warp', 
                random_state=42, 
                num_epochs=10, 
                num_threads=8,
                verbose=True):
    """
    Trains a LightFM model given the model parameters
    """
    model = LightFM(loss=loss, 
                    random_state=random_state, 
                    no_components=no_components, 
                    item_alpha=item_alpha, 
                    user_alpha=user_alpha)
    model.fit(train_sparse_dataset, 
              epochs=num_epochs, 
              verbose=verbose, 
              num_threads=num_threads)
    return model

def save_model(model, modelfilename):
    """
    Saves a LightFM model
    """
    pickle.dump(model, open(modelfilename, 'wb'))
    
def generate_recommendations(model_fn, save_fn, user_map_fn='BigDataProject/Maps/user_id_map.pkl', item_map_fn='BigDataProject/Maps/item_id_map.pkl'):
    """
    Generate Recommendations for a given model
    """
    recommendations = pd.DataFrame(columns=['user_id', 'recommendations'])
    
    #Loading files
    print('Loading Model and Map Files...')
    model = pickle.load(open(model_fn, 'rb'))
    user_map = pickle.load(open(user_map_fn, 'rb'))
    item_map = pickle.load(open(item_map_fn, 'rb'))
    
    user_map = {v: k for k, v in user_map.items()}
    item_map = {v: k for k, v in item_map.items()}
    
    #Getting user and item representations
    print('Getting User and Item Representations...')
    item_repr = model.get_item_representations()
    user_repr = model.get_user_representations()
    
    item_bias = item_repr[0]
    item_weights = item_repr[1]

    user_bias = user_repr[0]
    user_weights = user_repr[1]
    
    print("Generating Representations...")
    for i in tqdm(range(user_weights.shape[0])):
        user_embedding = user_weights[i]
        scores = user_embedding@item_weights.T
        scores += user_bias[i]
        scores = np.add(scores, item_bias)
        sorted_indices = np.argpartition(scores, -100)[-100:][::-1]
        user_id = user_map[i]
        item_mbids = [item_map[j] for j in sorted_indices]
        recommendations = recommendations.append({'user_id': user_id, 'recommendations': item_mbids}, ignore_index=True)
        
    print("Saving Recommendations...")
    recommendations.to_parquet(save_fn, engine='pyarrow')
    
def compute_metrics(validation_fn, recommendation_fn, k=100):
    spark = SparkSession.builder.config('spark.executor.memory', '70g').config('spark.driver.memory', '50g').config('spark.storage.memoryFraction', '0.05').appName("recsys").getOrCreate()
    val_df = val_df = spark.read.parquet(validation_fn, schema="recording_mbid INT, user_id INT, timestamp INT")
    reco_df = spark.read.parquet(recommendation_fn, schema="user_id INT, recommendations ARRAY")
    val_df = val_df.groupby("user_id").agg(collect_set("recording_mbid").alias('true_items'))
    reco_and_labels_df = reco_df.join(val_df, "user_id", "inner")
    reco_and_labels_rdd = reco_and_labels_df.rdd.map(lambda row: (row['recommendations'], row['true_items']))
    metrics = RankingMetrics(reco_and_labels_rdd)
    return metrics.precisionAt(k), metrics.meanAveragePrecisionAt(k), metrics.ndcgAt(k)

def run(train_fn,
        validation_fn,
        item_alpha=0.0, 
        user_alpha=0.0, 
        no_components=10, 
        loss='warp', 
        random_state=42, 
        num_epochs=10, 
        user_map = 'BigDataProject/Maps/user_id_map.pkl',
        item_map = 'BigDataProject/Maps/item_id_map.pkl',
        num_threads=8, 
        verbose=True):
    """
    Full Run
    """
    print('__________________________________________________\n')
    print(f'Item Alpha: {item_alpha}')
    print(f'User Alpha: {user_alpha}')
    print(f'Rank: {no_components}')
    print('Loading Train Data...')
    train_sparse_dataset = sparse.load_npz(train_fn)
    modelfilename = f'BigDataProject/Models/model_item_alpha_{item_alpha}_user_alpha_{user_alpha}_rank_{no_components}.pkl'
    recommendation_fn = f'BigDataProject/Recommendations/recommendation_item_alpha_{item_alpha}_user_alpha_{user_alpha}_rank_{no_components}.parquet'
    print('Beginning Training...')
    train_start = time.time()
    model = train_model(train_sparse_dataset, 
                item_alpha=item_alpha, 
                user_alpha=user_alpha, 
                no_components=no_components, 
                loss=loss, 
                random_state=random_state, 
                num_epochs=num_epochs, 
                num_threads=num_threads,
                verbose=verbose)
    end = time.time()
    time_taken_train = time.time() - train_start
    print(f'Time Taken For Training: {int(time_taken_train//60)} minutes and {int(time_taken_train - (time_taken_train//60)*60)} seconds')
    print('Saving Model...')
    save_model(model, modelfilename)
    print('Generating Recommendations...')
    recommendation_start = time.time()
    generate_recommendations(model_fn=modelfilename, save_fn=recommendation_fn)
    time_taken_recommendation = time.time() - recommendation_start
    print(f'Time Taken For Generating Recommendations: {int(time_taken_recommendation//60)} minutes and {int(time_taken_recommendation - (time_taken_recommendation//60)*60)} seconds')
    print('Computing Metrics...')
    precision, mapk, ndcg = compute_metrics(validation_fn, recommendation_fn)
    print(f'Precision@100: {precision}')
    print(f'MAP@100: {mapk}')
    print(f'ndcg@100: {ndcg}')
    return time_taken_train, time_taken_recommendation, precision, mapk, ndcg, modelfilename, recommendation_fn

user_alpha_list = [1e-6]
item_alpha_list = [1e-6]

result_df = pd.DataFrame(columns=['user_alpha', 'item_alpha', 'rank', 'train_time', 'modelfilename','recommendation_time', 'recommendationfilename', 'precision@100', 'map@100', 'ndcg@100'])
train_fn = 'BigDataProject/Datasets/train.npz'
val_fn = 'BigDataProject/Datasets/valData_final.parquet'

for ua in user_alpha_list:
    for ia in item_alpha_list:
        time_train, time_rec, p, m, n, model_fn, recommendation_fn = run(train_fn=train_fn, validation_fn=val_fn, user_alpha=ua, item_alpha=ia, no_components=25)
        result_df = result_df.append({
            'user_alpha': ua,
            'item_alpha': ia,
            'rank': 25,
            'train_time': time_train,
            'modelfilename': model_fn,
            'recommendation_time': time_rec,
            'recommendationfilename': recommendation_fn,
            'precision@100': p,
            'map@100': m,
            'ndcg@100': n
        }, ignore_index=True)

result_df.to_csv('BigDataProject/results_alpha_tuning.csv', index=False)