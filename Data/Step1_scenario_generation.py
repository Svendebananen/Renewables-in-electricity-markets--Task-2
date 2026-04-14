import os
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.cluster import KMeans # for clustering scenarios to make the sampled one more representative of the full dataset

SEED      = 42  # starting value for random number generation
N_WIND    = 20  # number of wind scenarios to sample
N_PRICE   = 20  # number of price scenarios to sample
N_IMBAL   = 4  # number of imbalance scenarios to sample
N_HOURS   = 24  # number of hours in a day
P_DEFICIT = 0.5 # probability of imbalance being a deficit (= being in a surplus)

DIR  = Path(__file__).parent # directory of this script
DATA  = DIR / "Raw"          # directory with raw data folder

# 1. LOAD & CLEAN PRICES
price_raw = pd.read_csv(DATA / "Price_DK2_2019.csv") # loads 2019 prices CSV into a DataFrame
price_raw['cet_cest_timestamp'] = pd.to_datetime(price_raw['cet_cest_timestamp'], utc=True) # convert cet time columns to a pandas datetime object
price_raw = price_raw[price_raw['cet_cest_timestamp'].dt.year == 2019] # filters rows keeping only those whose year in CET is 2019. 
price_raw['day']  = price_raw['cet_cest_timestamp'].dt.date # extracts the date (year-month-day) from the timestamp and stores it in a new column 'day'
price_raw['hour'] = price_raw['cet_cest_timestamp'].dt.hour # extracts the hour (0-23) from the timestamp and stores it in a new column 'hour'

# 2. LOAD & CLEAN WIND
wind_raw = pd.read_csv(DATA / "Wind_DK2_2019.csv", comment='#') # loads 2019 wind CSV into a DataFrame.
wind_raw['time'] = pd.to_datetime(wind_raw['time'], utc=True) # convert time columns to a pandas datetime object. 
wind_raw = wind_raw[wind_raw['time'].dt.year == 2019] # filters rows keeping only those whose year in UTC is 2019. 
wind_raw['day']  = wind_raw['time'].dt.date # extracts the date (year-month-day) from the timestamp and stores it in a new column 'day'
wind_raw['hour'] = wind_raw['time'].dt.hour # extracts the hour (0-23) from the timestamp and stores it in a new column 'hour'

# 3. RESHAPE TO MATRIX (days x hours)
price_matrix = price_raw.pivot(index='day', columns='hour', values='DK_2_price_day_ahead')
wind_matrix  = wind_raw.pivot(index='day',  columns='hour', values='electricity')
price_matrix = price_matrix.dropna() # removes any days with missing values
wind_matrix  = wind_matrix.dropna() # removes any days with missing values

# 4. CLUSTER PROFILES WITH K-MEANS
def cluster_profiles(matrix, n_clusters, seed):
    kmeans       = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)    # K-Means initialization (number of clusters, random seed, number of initializations)
    kmeans.fit(matrix) # exectute K-Means clustering on the input matrix; when converges, it assigns each day to a cluster and computes the centroids of the clusters (output = Numpy Array with the centroids)
    centroids    = pd.DataFrame(kmeans.cluster_centers_, columns=matrix.columns) # convert the Centroids Numpy Array to a DataFrame and add the name of the columns (hours) to the centroids DataFrame
    labels       = kmeans.labels_ # list of cluster labels assigned to each day
    cluster_size = np.bincount(labels) # counts the number of elements in each cluster (size of the cluster)
    probs        = cluster_size / len(matrix) # equal to the sum of the probabilities of the elements of the cluster
    return centroids, probs

price_centroids, price_probs = cluster_profiles(price_matrix, N_PRICE, SEED)
wind_centroids,  wind_probs  = cluster_profiles(wind_matrix,  N_WIND,  SEED)

price_centroids.to_csv(DIR / "Price_scenarios.csv", index_label='scenario_id')
wind_centroids.to_csv(DIR  / "Wind_scenarios.csv",  index_label='scenario_id')

# 5. GENERATE IMBALANCE SCENARIOS
np.random.seed(SEED)
si           = np.random.binomial(1, P_DEFICIT, size=(N_IMBAL, N_HOURS))
imbal_scen   = pd.DataFrame(si, columns=range(N_HOURS))
imbal_probs  = np.full(N_IMBAL, 1 / N_IMBAL)  # uniform probability
imbal_scen.to_csv(DIR / "Imbalance_scenarios.csv", index_label='scenario_id')

# 6. CARTESIAN PRODUCT
rows = []
for i_w, i_p, i_s in product(range(N_WIND), range(N_PRICE), range(N_IMBAL)):
    prob = wind_probs[i_w] * price_probs[i_p] * imbal_probs[i_s]
    for h in range(N_HOURS):
        wind   = wind_centroids.iloc[i_w, h]
        price  = price_centroids.iloc[i_p, h]
        si_val = imbal_scen.iloc[i_s, h]
        rows.append({
            'scenario_id': f"{i_w}_{i_p}_{i_s}",
            'hour':        h,
            'wind_mw':     wind,
            'da_price':    price,
            'si':          si_val,
            'prob':        prob,
        })

combined = pd.DataFrame(rows)
combined.to_csv(DIR / "Combined_scenarios.csv", index=False)
print(f"Generated scenarios: {combined['scenario_id'].nunique()}")