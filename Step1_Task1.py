import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 

DIR = Path(__file__).parent
DATA = DIR / "Data"
PLOTS = DIR / "Step 1 Plots"
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

# build model
model = gp.Model("task1_one_price")

# decision variables
p_DA = model.addVars(HOURS, lb=0, ub=P_NOM, name="p_DA") 

# compute balancing prices
lambda_B = lambda_DA.copy() # copy to initialize balancing price matrix with the same structure as lambda_DA
for ω in SCENARIOS:
    for h in HOURS:
        if si.loc[ω, h] == 1: # if deficit, set balancing price to 1.25 times the day-ahead price 
            lambda_B.loc[ω, h] = 1.25 * lambda_DA.loc[ω, h]
        else: # if surplus, set balancing price to 0.85 times the day-ahead price
            lambda_B.loc[ω, h] = 0.85 * lambda_DA.loc[ω, h]

# objective function (expected profit maximization)
model.setObjective(
    gp.quicksum(
        prob[ω] * (
            lambda_DA.loc[ω, h] * p_DA[h] +
            lambda_B.loc[ω, h]  * (wind_mw.loc[ω, h] - p_DA[h])
        )
        for ω in SCENARIOS
        for h in HOURS
    ),
    GRB.MAXIMIZE
) 

# constraints 
# p_DA = model.addVars(HOURS, lb=0, ub=P_NOM, name="p_DA"). USELESS since we already defined p_DA above with the same name and bouns, but left here for clarity.

# optimize model
model.optimize() 

# extract results
p_DA_values = {h: p_DA[h].X for h in HOURS}

# hourly expected profit
hourly_profit = {
    h: sum(
        prob[ω] * (
            lambda_DA.loc[ω, h] * p_DA_values[h] +
            lambda_B.loc[ω, h]  * (wind_mw.loc[ω, h] - p_DA_values[h])
        )
        for ω in SCENARIOS
    )
    for h in HOURS
}

total_profit = sum(hourly_profit.values())

print(f"Total expected profit: {total_profit:.2f} €")
print(f"Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")

# illustrate profit distribution across scenarios 
scenario_profit = {
    ω: sum(
        lambda_DA.loc[ω, h] * p_DA_values[h] +
        lambda_B.loc[ω, h]  * (wind_mw.loc[ω, h] - p_DA_values[h])
        for h in HOURS
    )
    for ω in SCENARIOS
} 
profits = list(scenario_profit.values())

plt.figure(figsize=(10, 5))
plt.hist(profits, bins=50, edgecolor='black')
plt.xlabel("Profit (€)")
plt.ylabel("Number of scenarios")
plt.title("Profit distribution across scenarios - One-price scheme")
plt.tight_layout()
plt.savefig(PLOTS / "Task1.1_profit_distribution.png", dpi=150)
plt.show()