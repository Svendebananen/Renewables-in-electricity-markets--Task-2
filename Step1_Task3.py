import gurobipy as gp 
import numpy as np
import pandas as pd
import pathlib as path
import matplotlib.pyplot as plt

DIR = path.Path(__file__).parent
DATA = DIR / "Data" 

# load combined scenarios
scenarios = pd.read_csv(DATA / "Combined_scenarios.csv")

# pivot to matrix format (scenarios x hours)
wind_mw    = scenarios.pivot(index='scenario_id', columns='hour', values='wind_mw') # pivot the 'wind_mw' column to create a matrix where rows are scenarios and columns are hours, with the values being the wind power in MW for each scenario and hour
lambda_DA  = scenarios.pivot(index='scenario_id', columns='hour', values='da_price') # pivot the 'da_price' column to create a matrix where rows are scenarios and columns are hours, with the values being the day-ahead price for each scenario and hour
si         = scenarios.pivot(index='scenario_id', columns='hour', values='si') # pivot the 'si' column to create a matrix where rows are scenarios and columns are hours, with the values being the imbalance indicator (1 for deficit, 0 for surplus) for each scenario and hour
prob       = scenarios.drop_duplicates('scenario_id').set_index('scenario_id')['prob'] # extract the probability of each scenario by dropping duplicate rows based on 'scenario_id', setting 'scenario_id' as the index, and selecting the 'prob' column

# parameters
P_NOM       = 500 # nominal power of the wind turbine in MW
N_HOURS     = 24  # number of hours in a day
HOURS       = range(N_HOURS) 
SCENARIOS = list(wind_mw.index) 

# compute balancing prices
lambda_B_up   = lambda_DA.copy()
lambda_B_down = lambda_DA.copy()

for omega in SCENARIOS:
    for h in HOURS:
        if si.loc[omega, h] == 1:  # upward need
            lambda_B_up.loc[omega, h] = lambda_DA.loc[omega, h]            # beneficial
            lambda_B_down.loc[omega, h] = 1.25 * lambda_DA.loc[omega, h]   # harmful
        else:                  # downward need
            lambda_B_up.loc[omega, h] = 0.85 * lambda_DA.loc[omega, h]     # harmful
            lambda_B_down.loc[omega, h] = lambda_DA.loc[omega, h]          # beneficial

def solve_two_price(scenarios_subset):
    
    model = gp.Model("two_price")
    model.setParam('OutputFlag', 0)
    
    p_DA       = model.addVars(HOURS, lb=0, ub=P_NOM, name="p_DA")
    delta_up   = model.addVars(scenarios_subset, HOURS, lb=0, ub=P_NOM, name="delta_up")
    delta_down = model.addVars(scenarios_subset, HOURS, lb=0, ub=P_NOM, name="delta_down")

    model.setObjective(
        gp.quicksum(
            prob[omega] * (
                lambda_DA.loc[omega, h] * p_DA[h] +
                lambda_B_up.loc[omega, h]   * delta_up[omega, h] -
                lambda_B_down.loc[omega, h] * delta_down[omega, h]
            )
            for omega in scenarios_subset
            for h in HOURS
        ),
        gp.GRB.MAXIMIZE
    )

    for omega in scenarios_subset:
        for h in HOURS:
            model.addConstr(delta_up[omega,h] - delta_down[omega,h] == wind_mw.loc[omega,h] - p_DA[h])

    model.optimize()
    
    p_DA_values = {h: p_DA[h].X for h in HOURS}
    return p_DA_values

# 8-fold cross-validation 
N_FOLDS    = 8    # number of folds for cross-validation
N_INSAMPLE = 200  # number of scenarios in the in-sample set for each fold


np.random.seed(42)
scenario_ids = np.array(SCENARIOS)
np.random.shuffle(scenario_ids)
folds = np.array_split(scenario_ids, N_FOLDS) # generate a list of 8 arrays, each containig 200 scenario ids

results = []

for fold_idx in range(N_FOLDS):
    insample  = folds[fold_idx]
    outsample = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_idx])
    prob_insample = prob[insample] / prob[insample].sum()
    prob_outsample = prob[outsample] / prob[outsample].sum()
    p_DA_values = solve_two_price(insample)

    # in-sample profit to compare with out-of-sample profit later
    insample_profit = sum(
    prob_insample[omega] * (
        lambda_DA.loc[omega, h] * p_DA_values[h] +
        lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_values[h], 0) -
        lambda_B_down.loc[omega, h] * max(p_DA_values[h] - wind_mw.loc[omega, h], 0)
    )
    for omega in insample
    for h in HOURS
    )

    # day-ahead profit (calculated on in-sample scenarios)
    da_profit = sum(
        prob_insample[omega] * lambda_DA.loc[omega, h] * p_DA_values[h]
        for omega in insample
        for h in HOURS
    )

    # out-of-sample imbalance cost
    outsample_average_imbalance_cost = sum(
    prob_outsample[omega] * (
        lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_values[h], 0) -
        lambda_B_down.loc[omega, h] * max(p_DA_values[h] - wind_mw.loc[omega, h], 0)
    )
    for omega in outsample
    for h in HOURS
    )

    # out-of-sample imbalance cost
    outsample_profit = da_profit + outsample_average_imbalance_cost

    results.append({
    'fold':              fold_idx,
    'insample_profit': insample_profit,
    'da_profit':   da_profit,
    'outsample_average_imbalance_cost':  outsample_average_imbalance_cost,
    'outsample_profit':  outsample_profit
    })

results_df = pd.DataFrame(results)

# comparison
avg_insample_da_profit  = results_df['da_profit'].mean()
avg_insample_profit = results_df['insample_profit'].mean()
avg_outsample_profit = results_df['outsample_profit'].mean()

print(f"Average out-of-sample profit: €{avg_outsample_profit:,.2f}")
print(f"Average in-sample profit:    €{avg_insample_profit:,.2f}")
print(f"Average in-sample DA profit: €{avg_insample_da_profit:,.2f}")
print()
print(results_df.to_string(index=False))
