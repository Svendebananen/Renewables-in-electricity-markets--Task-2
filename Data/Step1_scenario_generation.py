import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

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
price_matrix = price_matrix.dropna() # removes any days with missing values.
wind_matrix  = wind_matrix.dropna() # removes any days with missing values.

# 4. SAMPLE SYSTEM IMBALANCE PROFILES
np.random.seed(SEED)
price_sampled = price_matrix.sample(N_PRICE).reset_index(drop=True) #randomly selects N rows from the price matrix
wind_sampled  = wind_matrix.sample(N_WIND).reset_index(drop=True)# randomly selects N rows from the wind matrix

price_sampled.to_csv(DIR / "Price_scenarios.csv", index_label='scenario_id')
wind_sampled.to_csv(DIR / "Wind_scenarios.csv",   index_label='scenario_id')

# 5. GENERATE IMBALANCE SCENARIOS
si = np.random.binomial(1, P_DEFICIT, size=(N_IMBAL, N_HOURS)) #  draws random samples from a Bernoulli distribution.
imbal_df = pd.DataFrame(si, columns=range(N_HOURS))
imbal_df.to_csv(DIR / "Imbalance_scenarios.csv", index_label='scenario_id')

# 6. CARTESIAN PRODUCT
rows = []
for i_w, i_p, i_s in product(range(N_WIND), range(N_PRICE), range(N_IMBAL)):
    for h in range(N_HOURS):
        wind   = wind_sampled.iloc[i_w, h]
        price  = price_sampled.iloc[i_p, h]
        si_val = imbal_df.iloc[i_s, h]
        bp     = 1.25 * price if si_val == 1 else 0.85 * price
        rows.append({
            'scenario_id': f"{i_w}_{i_p}_{i_s}",
            'hour':        h,      # hour of the day 
            'wind_mw':     wind,   # wind power in MW
            'da_price':    price,  # day-ahead price
            'si':          si_val, # systema imbalance: 1 = deficit, 0 = surplus
        })

combined = pd.DataFrame(rows)
combined.to_csv(DIR / "Combined_scenarios.csv", index=False)
print(f"Generated scenarios: {combined['scenario_id'].nunique()}") # counts the number of unique values in scenario_id