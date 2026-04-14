import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent
DATA = DIR / "Data"

# load combined scenarios
scenarios = pd.read_csv(DATA / "Combined_scenarios.csv")

# pivot to matrix format (scenarios x hours)
wind_mw    = scenarios.pivot(index='scenario_id', columns='hour', values='wind_mw')
lambda_DA  = scenarios.pivot(index='scenario_id', columns='hour', values='da_price')
si         = scenarios.pivot(index='scenario_id', columns='hour', values='si')
prob       = scenarios.drop_duplicates('scenario_id').set_index('scenario_id')['prob']

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