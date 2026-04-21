import numpy as np
import pandas as pd
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS
)
from step1.models import compute_balancing_prices_two, solve_two_price
from step1.plots import plot_crossvalidation

# compute two-price balancing prices
lambda_B_up, lambda_B_down = compute_balancing_prices_two(lambda_DA, si)

# 8-fold cross-validation
N_FOLDS = 8

np.random.seed(42)
scenario_ids = np.array(SCENARIOS)
np.random.shuffle(scenario_ids)
folds = np.array_split(scenario_ids, N_FOLDS)

results = []

for fold_idx in range(N_FOLDS):
    insample   = folds[fold_idx]
    outsample  = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_idx])
    prob_insample  = prob[insample]  / prob[insample].sum()
    prob_outsample = prob[outsample] / prob[outsample].sum()

    p_DA_values, _, _ = solve_two_price(
        insample, prob_insample, wind_mw, lambda_DA, lambda_B_up, lambda_B_down
    )

    # in-sample profit
    insample_profit = sum(
        prob_insample[omega] * (
            lambda_DA.loc[omega, h]     * p_DA_values[h] +
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_values[h], 0) -
            lambda_B_down.loc[omega, h] * max(p_DA_values[h] - wind_mw.loc[omega, h], 0)
        )
        for omega in insample
        for h in HOURS
    )

    # day-ahead profit (in-sample)
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

    outsample_profit = da_profit + outsample_average_imbalance_cost

    results.append({
        'fold':                              fold_idx,
        'insample_profit':                   insample_profit,
        'da_profit':                         da_profit,
        'outsample_average_imbalance_cost':  outsample_average_imbalance_cost,
        'outsample_profit':                  outsample_profit,
    })

results_df = pd.DataFrame(results)

avg_insample_da_profit  = results_df['da_profit'].mean()
avg_insample_profit     = results_df['insample_profit'].mean()
avg_outsample_profit    = results_df['outsample_profit'].mean()

print(f"Average out-of-sample profit: {avg_outsample_profit:,.2f}€")
print(f"Average in-sample profit:    {avg_insample_profit:,.2f}€")
print(f"Average in-sample DA profit: {avg_insample_da_profit:,.2f}€")
print()
print(results_df.to_string(index=False))

plot_crossvalidation(
    results_df,
    save_path_scatter=PLOTS / "Task1.3_crossvalidation_scatter.png",
    save_path_bar=PLOTS    / "Task1.3_crossvalidation_barplot.png",
)
