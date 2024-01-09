import numpy as np
import pandas as pd
import dask.bag as db
from tqdm import tqdm
from annoy import AnnoyIndex

# Read ALS user and item latent representations
users = pd.read_parquet("/scratch/sn3250/userFactors_als20.parquet")
items = pd.read_parquet("/scratch/sn3250/itemFactors_als20.parquet")


# Create index and add items 
rank = 20
index = AnnoyIndex(rank, "dot")

def add_to_index(index, item):
    index.add_item(item[0], item[1]) # id, representation
    return None

items_bag = db.from_sequence(list(items.values))
items_bag.map(lambda item: add_to_index(index, item)).compute(scheduler='threads', num_workers=25)


# Build index
n_trees = 200    # Larger value gives more accurate results, but larger indexes
index.build(n_trees=n_trees, n_jobs=-1)
index.save(f"/scratch/sn3250/annoy_results/indices/item_index_{n_trees}.ann")


# Load index
index = AnnoyIndex(rank, "dot")
index.load(f"/scratch/sn3250/annoy_results/indices/item_index_{n_trees}.ann")


# Init ann_matches
search_k = -1
k_closest = 100
ann_matches = np.zeros((len(users), k_closest)) # Init empty matches


# Find top k nearest items for each user
for i in range(len(users)):
    ann_matches[i] = index.get_nns_by_vector(users['features'][i], k_closest, 
                                             search_k=search_k, include_distances=False)
        # search_k: run-time tradeoff between better accuracy and speed. Default value = n_trees * k_closest
        

# Create recommendation table for evaluation
recs = pd.DataFrame()
recs['user_id'] = np.repeat(users['id'], k_closest)
recs['recommendations'] = ann_matches.flatten()
recs['recommendations'] = recs['recommendations'].apply(int)
recs = recs.reset_index(drop=True)

# Export to parquet file
recs.to_parquet(f"/scratch/sn3250/annoy_results/matches/ann_recomm_ntrees{n_trees}_k{search_k}.parquet")