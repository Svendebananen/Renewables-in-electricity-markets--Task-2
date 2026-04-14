import gurobipy as gp 
import numpy as np
import pandas as pd
import pathlib as path
import matplotlib.pyplot as plt

DIR = path.Path(__file__).parent
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

# compute balancing prices
lambda_B_up   = lambda_DA.copy() # copy to initialize balancing price matrix with the same structure as lambda_DA
lambda_B_down = lambda_DA.copy() # copy to initialize balancing price matrix with the same structure as lambda_DA

for omega in SCENARIOS:
    for h in HOURS:
        if si.loc[omega, h] == 1:  # upward need
            lambda_B_up.loc[omega, h] = lambda_DA.loc[omega, h]            # beneficial
            lambda_B_down.loc[omega, h] = 1.25 * lambda_DA.loc[omega, h]   # harmful
        else:                  # downward need
            lambda_B_up.loc[omega, h] = 0.85 * lambda_DA.loc[omega, h]     # harmful
            lambda_B_down.loc[omega, h] = lambda_DA.loc[omega, h]          # beneficial

def solve_two_price(scenarios_subset,prob_subset): # defining the two-price solver function that will be used in each fold of the cross-validation
    
    model = gp.Model("two_price")
    model.setParam('OutputFlag', 0) # suppress Gurobi output for cleaner output during cross-validation
    
    p_DA       = model.addVars(HOURS, lb=0, ub=P_NOM, name="p_DA")
    delta_up   = model.addVars(scenarios_subset, HOURS, lb=0, ub=P_NOM, name="delta_up")
    delta_down = model.addVars(scenarios_subset, HOURS, lb=0, ub=P_NOM, name="delta_down")

    model.setObjective(
        gp.quicksum(
            prob_subset[omega] * (
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
N_INSAMPLE = 400  # number of scenarios in the in-sample set for each fold


np.random.seed(42)
scenario_ids = np.array(SCENARIOS)
np.random.shuffle(scenario_ids)
folds = np.array_split(scenario_ids, N_FOLDS) # generate a list of 8 arrays, each containig 200 scenario ids

results = []

for fold_idx in range(N_FOLDS):
    insample  = folds[fold_idx] # in-sample scenario ids for the current fold
    outsample = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_idx]) #out-of-sample scecario ids for the current fold
    prob_insample = prob[insample] / prob[insample].sum() # normalize probabilities for in-sample scenarios to sum to 1
    prob_outsample = prob[outsample] / prob[outsample].sum() # normalize probabilities for out-of-sample scenarios to sum to 1
    p_DA_values = solve_two_price(insample, prob_insample) # solve the two-price model using only the in-sample scenarios

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
    outsample_profit = da_profit + outsample_average_imbalance_cost # total out-of-sample profit is the sum of day-ahead profit and imbalance cost (which can be positive or negative)

    results.append({
    'fold':              fold_idx,
    'insample_profit': insample_profit,
    'da_profit':   da_profit,
    'outsample_average_imbalance_cost':  outsample_average_imbalance_cost,
    'outsample_profit':  outsample_profit
    })

results_df = pd.DataFrame(results)

# comparison
avg_insample_da_profit  = results_df['da_profit'].mean()      # average day-ahead profit across folds
avg_insample_profit = results_df['insample_profit'].mean()    # average in-sample profit across folds
avg_outsample_profit = results_df['outsample_profit'].mean()  # average out-of-sample profit across folds (ex-post profit of K-fold cross validation)

print(f"Average out-of-sample profit: {avg_outsample_profit:,.2f}€")
print(f"Average in-sample profit:    {avg_insample_profit:,.2f}€")
print(f"Average in-sample DA profit: {avg_insample_da_profit:,.2f}€")
print()
print(results_df.to_string(index=False))


# scatter plot of in-sample vs out-of-sample profit for each fold
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(results_df['insample_profit'], results_df['outsample_profit'], 
           color='steelblue', s=100, zorder=5)

# diagonal reference line
min_val = min(results_df['insample_profit'].min(), results_df['outsample_profit'].min())
max_val = max(results_df['insample_profit'].max(), results_df['outsample_profit'].max())
margin = (max_val - min_val) * 0.5
ax.plot([min_val - margin * 0.7, max_val + margin * 0.7], 
        [min_val - margin * 0.7, max_val + margin * 0.7], 
        'r--', linewidth=1.5, label='Perfect generalization')

# fold labels
for i, row in results_df.iterrows():
    ax.annotate(f"Fold {row['fold']}", (row['insample_profit'], row['outsample_profit']),
                textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(min_val - margin, max_val + margin)
ax.set_ylim(min_val - margin, max_val + margin)
ax.set_xlabel("In-sample profit (€)")
ax.set_ylabel("Out-of-sample profit (€)")
ax.set_title("In-sample vs Out-of-sample profit - 8-fold cross-validation")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "Task1.3_crossvalidation_scatter.png", dpi=150)
plt.show()

# bar plot
x = np.arange(N_FOLDS)
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, results_df['insample_profit'],  width, label='In-sample',  color='steelblue')
ax.bar(x + width/2, results_df['outsample_profit'], width, label='Out-of-sample', color='orange')

ax.set_xlabel('Fold')
ax.set_ylabel('Profit (€)')
ax.set_title('In-sample vs Out-of-sample profit per fold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in range(N_FOLDS)])
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "Task1.3_crossvalidation_barplot.png", dpi=150)
plt.show()
