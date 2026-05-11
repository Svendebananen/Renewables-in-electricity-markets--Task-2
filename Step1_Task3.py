"""
Task 1.3) Ex-post Analysis
8-fold cross validation using 200 in-sample and 1400 out-of-sample scenarios.
Two-price and one-price schemes.
"""

import numpy as np
import pandas as pd
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS
)
from step1.models import (
    compute_balancing_prices_one, compute_balancing_prices_two,
    solve_one_price, solve_two_price
)
from step1.plots import plot_crossvalidation

lambda_B                   = compute_balancing_prices_one(lambda_DA, si)
lambda_B_up, lambda_B_down = compute_balancing_prices_two(lambda_DA, si)

N_FOLDS = 8
np.random.seed(42)
scenario_ids = np.array(SCENARIOS)
np.random.shuffle(scenario_ids)
folds = np.array_split(scenario_ids, N_FOLDS)

results_two = []
results_one = []

for fold_idx in range(N_FOLDS):
    insample  = folds[fold_idx]
    outsample = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_idx])
    prob_in  = prob[insample]  / prob[insample].sum()
    prob_out = prob[outsample] / prob[outsample].sum()

    # -------- Two-price scheme --------
    p_DA_two, _, _, _, _ = solve_two_price(
        insample, prob_in, wind_mw, lambda_DA, lambda_B_up, lambda_B_down
    )

    insample_profit_two = sum(
        prob_in[omega] * (
            lambda_DA.loc[omega, h]     * p_DA_two[h] +
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_two[h], 0) -
            lambda_B_down.loc[omega, h] * max(p_DA_two[h] - wind_mw.loc[omega, h], 0)
        )
        for omega in insample for h in HOURS
    )
    da_profit_two = sum(
        prob_in[omega] * lambda_DA.loc[omega, h] * p_DA_two[h]
        for omega in insample for h in HOURS
    )
    outsample_bal_two = sum(
        prob_out[omega] * (
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_two[h], 0) -
            lambda_B_down.loc[omega, h] * max(p_DA_two[h] - wind_mw.loc[omega, h], 0)
        )
        for omega in outsample for h in HOURS
    )
    results_two.append({
        'fold':             fold_idx,
        'insample_profit':  insample_profit_two,
        'da_profit':        da_profit_two,
        'outsample_profit': da_profit_two + outsample_bal_two,
    })

    # -------- One-price scheme --------
    p_DA_one, _, _, _, _ = solve_one_price(
        insample, prob_in, wind_mw, lambda_DA, lambda_B
    )

    insample_profit_one = sum(
        prob_in[omega] * (
            lambda_DA.loc[omega, h] * p_DA_one[h] +
            lambda_B.loc[omega, h]  * (wind_mw.loc[omega, h] - p_DA_one[h])
        )
        for omega in insample for h in HOURS
    )
    da_profit_one = sum(
        prob_in[omega] * lambda_DA.loc[omega, h] * p_DA_one[h]
        for omega in insample for h in HOURS
    )
    outsample_bal_one = sum(
        prob_out[omega] * lambda_B.loc[omega, h] * (wind_mw.loc[omega, h] - p_DA_one[h])
        for omega in outsample for h in HOURS
    )
    results_one.append({
        'fold':             fold_idx,
        'insample_profit':  insample_profit_one,
        'da_profit':        da_profit_one,
        'outsample_profit': da_profit_one + outsample_bal_one,
    })

df_two = pd.DataFrame(results_two)
df_one = pd.DataFrame(results_one)

for label, df in [("Two-price", df_two), ("One-price", df_one)]:
    print(f"\n=== {label} cross-validation ===")
    print(f"Average in-sample profit:  {df['insample_profit'].mean():,.2f}€")
    print(f"Average out-of-sample profit: {df['outsample_profit'].mean():,.2f}€")
    print(df.to_string(index=False))

plot_crossvalidation(
    df_two,
    save_path_scatter=PLOTS / "Task1.3_two_price_crossvalidation_scatter.png",
    save_path_bar=    PLOTS / "Task1.3_two_price_crossvalidation_barplot.png",
)
plot_crossvalidation(
    df_one,
    save_path_scatter=PLOTS / "Task1.3_one_price_crossvalidation_scatter.png",
    save_path_bar=    PLOTS / "Task1.3_one_price_crossvalidation_barplot.png",
)