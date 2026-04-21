import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 

DIR = Path(__file__).parent
DATA = DIR / "Data"
PLOTS = DIR / "Step 1 Plots"
PLOTS.mkdir(parents=True, exist_ok=True)
# load combined scenarios
scenarios = pd.read_csv(DATA / "Combined_scenarios.csv")

# pivot to matrix format (scenarios x hours)
wind_mw    = scenarios.pivot(index='scenario_id', columns='hour', values='wind_mw') # pivot the 'wind_mw' column to create a matrix where rows are scenarios and columns are hours, with the values being the wind power in MW for each scenario and hour
lambda_DA  = scenarios.pivot(index='scenario_id', columns='hour', values='da_price') # pivot the 'da_price' column to create a matrix where rows are scenarios and columns are hours, with the values being the day-ahead price for each scenario and hour
si         = scenarios.pivot(index='scenario_id', columns='hour', values='si') # pivot the system imbalance 'si' column to create a matrix where rows are scenarios and columns are hours, with the values being the imbalance indicator (1 for deficit, 0 for surplus) for each scenario and hour
prob       = scenarios.drop_duplicates('scenario_id').set_index('scenario_id')['prob'] # extract the probability of each scenario by dropping duplicate rows based on 'scenario_id', setting 'scenario_id' as the index, and selecting the 'prob' column

# parameters
P_NOM       = 500 # nominal power of the wind turbine in MW
N_HOURS     = 24  # number of hours in a day
HOURS       = range(N_HOURS) 
SCENARIOS = list(wind_mw.index)
DEFICIT_MULTIPLIER = 1.25 # multiplier for balancing price in deficit scenarios
SURPLUS_MULTIPLIER = 0.85 # multiplier for balancing price in surplus scenarios
ALPHA = 0.9 # confidence level for CVaR calculation

# build model
model = gp.Model("task1_one_price")

# decision variables
p_DA = model.addVars(HOURS, lb=0, ub=P_NOM, name="p_DA") 
Aux = model.addVars(SCENARIOS, lb=0, ub=GRB.INFINITY, name="Aux") # CVaR shortfall variable
Etha = model.addVar(lb=-GRB.INFINITY, name="Etha") # auxiliary variable for CVaR calculation
beta = 0 # weight of CVaR in the objective function (0 for pure expected profit maximization, 1 for pure CVaR minimization)

# compute balancing prices
lambda_B = lambda_DA.copy() # copy to initialize balancing price matrix with the same structure as lambda_DA
for omega in SCENARIOS:
    for h in HOURS:
        if si.loc[omega, h] == 1: # if deficit, set balancing price to 1.25 times the day-ahead price 
            lambda_B.loc[omega, h] = DEFICIT_MULTIPLIER * lambda_DA.loc[omega, h]
        else: # if surplus, set balancing price to 0.85 times the day-ahead price
            lambda_B.loc[omega, h] = SURPLUS_MULTIPLIER * lambda_DA.loc[omega, h]

# objective function (expected profit + optional CVaR term)
expected_profit_expr = gp.quicksum(
    prob[omega] * (
        lambda_DA.loc[omega, h] * p_DA[h] +
        lambda_B.loc[omega, h] * (wind_mw.loc[omega, h] - p_DA[h])
    )
    for omega in SCENARIOS
    for h in HOURS
)

cvar_expr = Etha - (1 / (1 - ALPHA)) * gp.quicksum(
    prob[omega] * Aux[omega]
    for omega in SCENARIOS
)

model.setObjective((1 - beta) * expected_profit_expr + beta * cvar_expr, GRB.MAXIMIZE)

# CVaR linking constraints: Aux[omega] >= eta - profit_in_scenario_omega
scenario_profit_expr = {
    omega: gp.quicksum(
        lambda_DA.loc[omega, h] * p_DA[h] +
        lambda_B.loc[omega, h] * (wind_mw.loc[omega, h] - p_DA[h])
        for h in HOURS
    )
    for omega in SCENARIOS
}

model.addConstrs(
    (Aux[omega] >= Etha - scenario_profit_expr[omega] for omega in SCENARIOS),
    name="cvar_shortfall"
)

# constraints 
# p_DA = model.addVars(HOURS, lb=0, ub=P_NOM, name="p_DA"). USELESS since we already defined p_DA above with the same name and bouns, but left here for clarity.

# run a beta sweep to show expected profit vs CVaR trade-off
beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
frontier = []
baseline = None

for beta in beta_values:
    model.setObjective((1 - beta) * expected_profit_expr + beta * cvar_expr, GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        continue

    p_DA_values = {h: p_DA[h].X for h in HOURS}
    scenario_profit = {
        omega: sum(
            lambda_DA.loc[omega, h] * p_DA_values[h] +
            lambda_B.loc[omega, h] * (wind_mw.loc[omega, h] - p_DA_values[h])
            for h in HOURS
        )
        for omega in SCENARIOS
    }

    expected_profit = sum(prob[omega] * scenario_profit[omega] for omega in SCENARIOS)
    cvar_value = Etha.X - (1 / (1 - ALPHA)) * sum(prob[omega] * Aux[omega].X for omega in SCENARIOS)

    frontier.append({
        "beta": beta,
        "expected_profit": expected_profit,
        "cvar": cvar_value,
        "p_DA_values": p_DA_values,
        "scenario_profit": scenario_profit
    })

    if abs(beta) < 1e-12:
        baseline = frontier[-1]

if baseline is None:
    raise RuntimeError("No optimal solution found for beta = 0.")

# print baseline (beta = 0) results
p_DA_values = baseline["p_DA_values"]
scenario_profit = baseline["scenario_profit"]

hourly_profit = {
    h: sum(
        prob[omega] * (
            lambda_DA.loc[omega, h] * p_DA_values[h] +
            lambda_B.loc[omega, h] * (wind_mw.loc[omega, h] - p_DA_values[h])
        )
        for omega in SCENARIOS
    )
    for h in HOURS
}

total_profit = baseline["expected_profit"]

print(f"Total expected profit: {total_profit:.2f} €")
print("Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")

print("\nBeta sweep (expected profit vs CVaR):")
for row in frontier:
    print(f"  beta={row['beta']:.2f} | E[profit]={row['expected_profit']:.2f} | CVaR={row['cvar']:.2f}")

# Addressing "illustrate profit distribution across scenarios"
profits = list(scenario_profit.values())
profits_array = np.array(profits)

# statistical analysis of scenarios
min_val = profits_array.min()
max_val = profits_array.max()
range_val = max_val - min_val

print()
print(f"Range: €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range: {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

# plot baseline scenario distribution (beta = 0)
plt.figure(figsize=(10, 5))
plt.hist(profits, bins=50, color="#fa9537", edgecolor="white")
plt.axvline(x=total_profit, color="red", linestyle="--", linewidth=2, label=f" Total expected profit: €{total_profit:,.0f}")
plt.legend()
plt.xlabel("Profit (€)")
plt.ylabel("Number of scenarios")
plt.title("Profit distribution across scenarios - One-price scheme (beta = 0)")
plt.tight_layout()
plt.savefig(PLOTS / "Task1.4_profit_distribution.png", dpi=150)
plt.show()

# plot profit-CVaR trade-off
frontier_df = pd.DataFrame(frontier)
plt.figure(figsize=(10, 5))
plt.plot(frontier_df["cvar"], frontier_df["expected_profit"], marker="o", color="#1f77b4")
for _, row in frontier_df.iterrows():
    plt.annotate(f"beta={row['beta']:.2f}", (row["cvar"], row["expected_profit"]), textcoords="offset points", xytext=(5, 5))
plt.xlabel("CVaR (alpha = 0.90)")
plt.ylabel("Expected Profit (€)")
plt.title("Expected Profit vs CVaR Trade-off")
plt.tight_layout()
plt.savefig(PLOTS / "Task1.4_profit_cvar_tradeoff.png", dpi=150)
plt.show()