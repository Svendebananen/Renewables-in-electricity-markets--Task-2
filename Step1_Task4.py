"""
Task 1.4) Risk-Averse Offering Strategy
"""

import time
import math
import pandas as pd
import numpy as np
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS, OUTPUTS
)
from step1.models import (
    compute_balancing_prices_one, compute_balancing_prices_two,
    solve_one_price, solve_two_price
)
from step1.plots import (
    plot_cvar_frontier_With_Both_Models, plot_profit_histogram,
    plot_cvar_frontier, plot_profit_boxplot, plot_profit_boxplot_comparison,
    plot_hourly_offers_frontier, plot_imbalance_transition
)

ALPHA       = 0.9
BETA_VALUES = [0.0, 0.02, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Balancing prices
# ---------------------------------------------------------------------------
lambda_B                   = compute_balancing_prices_one(lambda_DA, si)
lambda_B_up, lambda_B_down = compute_balancing_prices_two(lambda_DA, si)

# ---------------------------------------------------------------------------
# Beta sweep - one-price model
# ---------------------------------------------------------------------------
print("=== One-price model - beta sweep ===")
frontier_one = []
baseline_one = None

for beta in BETA_VALUES:
    t0 = time.perf_counter()
    p_DA_values, scenario_profit, cvar_value, _, _ = solve_one_price(
        SCENARIOS, prob, wind_mw, lambda_DA, lambda_B, beta=beta, alpha=ALPHA
    )
    solve_time      = time.perf_counter() - t0
    expected_profit = sum(prob[omega] * scenario_profit[omega] for omega in SCENARIOS)
    frontier_one.append({
        "beta":           beta,
        "expected_profit": expected_profit,
        "cvar":           cvar_value,
        "solve_time":     solve_time,
        "p_DA_values":    p_DA_values,
        "scenario_profit": scenario_profit,
    })
    if abs(beta) < 1e-12:
        baseline_one = frontier_one[-1]

pd.DataFrame(frontier_one).drop(columns=["p_DA_values", "scenario_profit"]).to_csv(
    OUTPUTS / "1.4_frontier_one.csv", index=False
)

# baseline stats (beta=0)
p_DA_values     = baseline_one["p_DA_values"]
scenario_profit = baseline_one["scenario_profit"]
total_profit    = baseline_one["expected_profit"]

hourly_profit = {
    h: sum(
        prob[omega] * (
            lambda_DA.loc[omega, h] * p_DA_values[h] +
            lambda_B.loc[omega, h]  * (wind_mw.loc[omega, h] - p_DA_values[h])
        )
        for omega in SCENARIOS
    )
    for h in HOURS
}

print(f"\nTotal expected profit (beta=0): {total_profit:,.2f} €\n")
print("Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")

profits_array = np.array(list(scenario_profit.values()))
min_val   = profits_array.min()
max_val   = profits_array.max()
range_val = max_val - min_val
print()
print(f"Range:              €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range:    {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

# gradient analysis - one-price
rows_grad_one = []
for entry in frontier_one:
    beta         = entry["beta"]
    profit_dict  = entry["scenario_profit"]
    p_DA_opt     = entry["p_DA_values"]
    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]
    for h in HOURS:
        c         = {omega: lambda_DA.loc[omega, h] - lambda_B.loc[omega, h] for omega in SCENARIOS}
        grad_E    = sum(prob[omega] * c[omega] for omega in SCENARIOS)
        grad_cvar = (1 / (1 - ALPHA)) * sum(prob[omega] * c[omega] for omega in worst_omegas)
        rows_grad_one.append({
            "beta":      beta,
            "hour":      h + 1,
            "grad_E":    grad_E,
            "grad_CVaR": grad_cvar,
            "same_sign": (grad_E * grad_cvar) >= 0,
            "p_DA":      p_DA_opt[h],
        })
pd.DataFrame(rows_grad_one).to_csv(OUTPUTS / "1.4_gradient_analysis_one.csv", index=False)

# ---------------------------------------------------------------------------
# Beta sweep - two-price model
# ---------------------------------------------------------------------------
print("\n=== Two-price model - beta sweep ===")
frontier_two = []
baseline_two = None

for beta in BETA_VALUES:
    t0 = time.perf_counter()
    p_DA_values, scenario_profit, cvar_value, _, _ = solve_two_price(
        SCENARIOS, prob, wind_mw, lambda_DA, lambda_B_up, lambda_B_down,
        beta=beta, alpha=ALPHA
    )
    solve_time      = time.perf_counter() - t0
    expected_profit = sum(prob[omega] * scenario_profit[omega] for omega in SCENARIOS)
    print(f"  beta={beta:.2f} | E[profit]={expected_profit:,.2f} | CVaR={cvar_value:,.2f} | time={solve_time:.3f}s")
    frontier_two.append({
        "beta":            beta,
        "expected_profit": expected_profit,
        "cvar":            cvar_value,
        "solve_time":      solve_time,
        "p_DA_values":     p_DA_values,
        "scenario_profit": scenario_profit,
    })
    if abs(beta) < 1e-12:
        baseline_two = frontier_two[-1]

pd.DataFrame(frontier_two).drop(columns=["p_DA_values", "scenario_profit"]).to_csv(
    OUTPUTS / "1.4_frontier_two.csv", index=False
)

# baseline stats (beta=0)
p_DA_values     = baseline_two["p_DA_values"]
scenario_profit = baseline_two["scenario_profit"]
total_profit    = baseline_two["expected_profit"]

hourly_profit = {
    h: sum(
        prob[omega] * (
            lambda_DA.loc[omega, h]     * p_DA_values[h] +
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_values[h], 0) -
            lambda_B_down.loc[omega, h] * max(p_DA_values[h] - wind_mw.loc[omega, h], 0)
        )
        for omega in SCENARIOS
    )
    for h in HOURS
}

print(f"\nTotal expected profit (beta=0): {total_profit:.2f} €")
print("Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")

profits_array = np.array(list(scenario_profit.values()))
min_val   = profits_array.min()
max_val   = profits_array.max()
range_val = max_val - min_val
print()
print(f"Range:              €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range:    {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

# profit decomposition vs beta (two-price)
rows_decomp = []
for entry in frontier_two:
    beta         = entry["beta"]
    profit_dict  = entry["scenario_profit"]
    p_DA_opt     = entry["p_DA_values"]
    profits      = np.array(list(profit_dict.values()))
    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]
    da_list, bal_list = [], []
    for omega in worst_omegas:
        da  = sum(lambda_DA.loc[omega, h] * p_DA_opt[h] for h in HOURS)
        bal = sum(
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_opt[h], 0)
            - lambda_B_down.loc[omega, h] * max(p_DA_opt[h] - wind_mw.loc[omega, h], 0)
            for h in HOURS
        )
        da_list.append(da)
        bal_list.append(bal)
    rows_decomp.append({
        "beta":            beta,
        "E[profit]":       sum(prob[omega] * profit_dict[omega] for omega in SCENARIOS),
        "std":             np.std(profits),
        "min":             profits.min(),
        "CVaR":            entry["cvar"],
        "E[profit|worst]": np.mean([d + b for d, b in zip(da_list, bal_list)]),
        "E[DA|worst]":     np.mean(da_list),
        "E[bal|worst]":    np.mean(bal_list),
    })

df_table = pd.DataFrame(rows_decomp)
df_table[["E[profit]", "std", "min", "CVaR",
          "E[profit|worst]", "E[DA|worst]", "E[bal|worst]"]] = \
    df_table[["E[profit]", "std", "min", "CVaR",
              "E[profit|worst]", "E[DA|worst]", "E[bal|worst]"]].round(0).astype(int)
df_table.to_csv(OUTPUTS / "1.4_profit_decomposition_two.csv", index=False)

# penalized imbalance structure vs beta (two-price)
rows_imbal = []
for entry in frontier_two:
    beta         = entry["beta"]
    profit_dict  = entry["scenario_profit"]
    p_DA_opt     = entry["p_DA_values"]
    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]
    total  = n_worst * len(HOURS)
    n_def  = sum(
        1 for omega in worst_omegas for h in HOURS
        if wind_mw.loc[omega, h] < p_DA_opt[h] and si.loc[omega, h] == 1
    )
    n_sur  = sum(
        1 for omega in worst_omegas for h in HOURS
        if wind_mw.loc[omega, h] > p_DA_opt[h] and si.loc[omega, h] == 0
    )
    avg_imbal = np.mean([
        wind_mw.loc[omega, h] - p_DA_opt[h]
        for omega in worst_omegas for h in HOURS
    ])
    rows_imbal.append({
        "beta":         beta,
        "deficit_pen%": round(100 * n_def / total, 1),
        "surplus_pen%": round(100 * n_sur / total, 1),
        "avg_imbal_MW": round(avg_imbal, 1),
    })
pd.DataFrame(rows_imbal).to_csv(OUTPUTS / "1.4_penalized_imbalance.csv", index=False)

# gradient analysis - two-price
rows_grad_two = []
for entry in frontier_two:
    beta         = entry["beta"]
    profit_dict  = entry["scenario_profit"]
    p_DA_opt     = entry["p_DA_values"]
    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]
    for h in HOURS:
        def grad_profit(omega, h=h):
            w = wind_mw.loc[omega, h]
            p = p_DA_opt[h]
            if w < p:   return lambda_DA.loc[omega, h] - lambda_B_down.loc[omega, h]
            elif w > p: return lambda_DA.loc[omega, h] - lambda_B_up.loc[omega, h]
            else:       return lambda_DA.loc[omega, h]
        grad_E    = sum(prob[omega] * grad_profit(omega) for omega in SCENARIOS)
        grad_cvar = (1 / (1 - ALPHA)) * sum(prob[omega] * grad_profit(omega) for omega in worst_omegas)
        rows_grad_two.append({
            "beta":      beta,
            "hour":      h + 1,
            "grad_E":    grad_E,
            "grad_CVaR": grad_cvar,
            "same_sign": (grad_E * grad_cvar) >= 0,
            "p_DA":      p_DA_opt[h],
        })
pd.DataFrame(rows_grad_two).to_csv(OUTPUTS / "1.4_gradient_analysis_two.csv", index=False)

# worst-scenario analysis (two-price, beta=0 and beta=1)
def worst_scenario_analysis(frontier_entry, label):
    profit_dict  = frontier_entry["scenario_profit"]
    p_DA_opt     = frontier_entry["p_DA_values"]
    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]
    rows = []
    for omega in worst_omegas:
        for h in HOURS:
            w     = wind_mw.loc[omega, h]
            p     = p_DA_opt[h]
            imbal = w - p
            si_h  = si.loc[omega, h]
            penalized = (si_h == 1) if imbal < 0 else (si_h == 0)
            rows.append({
                "omega": omega, "hour": h, "profit": profit_dict[omega],
                "imbal_mw": imbal, "si": si_h, "penalized": penalized,
                "lda": lambda_DA.loc[omega, h],
            })
    df    = pd.DataFrame(rows)
    total = len(df)
    n_pen = df["penalized"].sum()
    n_def = ((df["imbal_mw"] < 0) & df["penalized"]).sum()
    n_sur = ((df["imbal_mw"] > 0) & df["penalized"]).sum()
    print(f"\n=== Worst-scenario analysis - two-price, {label}, alpha={ALPHA} ===")
    print(f"Scenarios analysed:  {n_worst} / {len(SCENARIOS)}")
    print(f"Total hour-slots:    {total}")
    print(f"Penalized slots:     {n_pen}  ({100*n_pen/total:.1f}%)")
    print(f"  deficit (up-reg):  {n_def}  ({100*n_def/total:.1f}%)")
    print(f"  surplus (down-reg):{n_sur}  ({100*n_sur/total:.1f}%)")
    print(f"Avg imbalance [MW]:  {df['imbal_mw'].mean():.1f}")
    print(f"Avg DA price [€/MWh]:{df['lda'].mean():.1f}")
    return df

worst_scenario_analysis(baseline_two,     "beta=0")
worst_scenario_analysis(frontier_two[-1], "beta=1")

print("\n=== Solve times summary ===")
print(f"{'model':>12} {'beta':>6} {'time (s)':>10}")
for entry in frontier_one:
    print(f"{'one-price':>12} {entry['beta']:>6.2f} {entry['solve_time']:>10.3f}")
for entry in frontier_two:
    print(f"{'two-price':>12} {entry['beta']:>6.2f} {entry['solve_time']:>10.3f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
frontier_one_df = pd.DataFrame(frontier_one)
frontier_two_df = pd.DataFrame(frontier_two)

plot_profit_histogram(
    baseline_one["scenario_profit"], prob,
    title="Profit distribution - One-price scheme (beta=0)",
    save_path=PLOTS / "Task1.4_one_price_profit_distribution.png",
    color="#fa9537",
)
plot_profit_histogram(
    frontier_two[-1]["scenario_profit"], prob,
    save_path=PLOTS / "Task1.4_two_price_profit_distribution_beta1.png",
    color="#4CAF50",
)
plot_cvar_frontier(
    frontier_one_df,
    title="Expected Profit vs CVaR - One-price scheme",
    save_path=PLOTS / "Task1.4_one_price_profit_cvar_tradeoff.png",
)
plot_cvar_frontier(
    frontier_two_df,
    title="Expected Profit vs CVaR - Two-price scheme",
    save_path=PLOTS / "Task1.4_two_price_profit_cvar_tradeoff.png",
)
plot_cvar_frontier_With_Both_Models(
    frontier_one_df, frontier_two_df, None,
    PLOTS / "Task1.4_both_models_profit_cvar_tradeoff.png",
)
plot_profit_boxplot(
    frontier_one, color="#fa9537",
    title="Profit distribution vs. risk aversion - one-price",
    save_path=PLOTS / "Task1.4_one_price_boxplot.png",
)
plot_profit_boxplot(
    frontier_two, color="#3fe60c",
    title="Profit distribution vs. risk aversion - two-price",
    save_path=PLOTS / "Task1.4_two_price_boxplot.png",
)
plot_profit_boxplot_comparison(
    frontier_one, frontier_two,
    save_path=PLOTS / "Task1.4_boxplot_comparison.png",
)
plot_hourly_offers_frontier(
    frontier_two, wind_mw, si, SCENARIOS,
    save_path=PLOTS / "Task1.4_two_price_hourly_offers_diff.png",
)
plot_imbalance_transition(
    frontier_two, wind_mw, si, ALPHA, HOURS, SCENARIOS,
    save_path=PLOTS / "Task1.4_imbalance_transition.png",
)