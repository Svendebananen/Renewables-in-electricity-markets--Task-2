"""
Task 1.4) Risk-Averse Offering Strategy
"""

import pandas as pd
import numpy as np
import math
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS
)
from step1.models import (
    compute_balancing_prices_one, compute_balancing_prices_two,
    solve_one_price, solve_two_price
)
from step1.plots import plot_cvar_frontier_With_Both_Models, plot_profit_histogram, plot_cvar_frontier, plot_profit_boxplot, plot_profit_boxplot_comparison,plot_hourly_offers, plot_imbalance_transition

ALPHA       = 0.9
BETA_VALUES = [0.0, 0.02, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Balancing prices
# ---------------------------------------------------------------------------
lambda_B                    = compute_balancing_prices_one(lambda_DA, si)
lambda_B_up, lambda_B_down  = compute_balancing_prices_two(lambda_DA, si)

# ---------------------------------------------------------------------------
# Beta sweep – one-price model
# ---------------------------------------------------------------------------
print("=== One-price model – beta sweep ===")
frontier_one = []
baseline_one = None

for beta in BETA_VALUES:
    p_DA_values, scenario_profit, cvar_value, _, _ = solve_one_price(
        SCENARIOS, prob, wind_mw, lambda_DA, lambda_B, beta=beta, alpha=ALPHA
    )
    expected_profit = sum(prob[omega] * scenario_profit[omega] for omega in SCENARIOS)

    p_DA_str = "  ".join(f"h{h}:{p_DA_values[h]:.1f}" for h in HOURS)
    print(f"  beta={beta:.2f} | E[profit]={expected_profit:,.2f} | CVaR={cvar_value:,.2f} | {p_DA_str}")

    frontier_one.append({
        "beta": beta,
        "expected_profit": expected_profit,
        "cvar": cvar_value,
        "p_DA_values": p_DA_values,
        "scenario_profit": scenario_profit,
    })
    if abs(beta) < 1e-12:
        baseline_one = frontier_one[-1]

print("Beta sweep (expected profit vs CVaR):")
for row in frontier_one:
    print(f"  beta={row['beta']:.2f} | E[profit]={row['expected_profit']:.2f} | CVaR={row['cvar']:.2f}")

# print baseline stats (beta = 0)
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
print(f"Range: €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range: {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

plot_profit_histogram(
    scenario_profit, prob,
    title="Profit distribution across scenarios - One-price scheme (beta = 0)",
    save_path=PLOTS / "Task1.4_one_price_profit_distribution.png",
    color="#fa9537",
)

frontier_one_df = pd.DataFrame(frontier_one)
plot_cvar_frontier(
    frontier_one_df,
    title="Expected Profit vs CVaR Trade-off - One-price scheme",
    save_path=PLOTS / "Task1.4_one_price_profit_cvar_tradeoff.png",
)

# ---------------------------------------------------------------------------
# Beta sweep – two-price model
# ---------------------------------------------------------------------------
print("\n=== Two-price model – beta sweep ===")
frontier_two = []
baseline_two = None

for beta in BETA_VALUES:
    p_DA_values, scenario_profit, cvar_value, _, _ = solve_two_price(
        SCENARIOS, prob, wind_mw, lambda_DA, lambda_B_up, lambda_B_down,
        beta=beta, alpha=ALPHA
    )
    expected_profit = sum(prob[omega] * scenario_profit[omega] for omega in SCENARIOS)
    frontier_two.append({
        "beta": beta,
        "expected_profit": expected_profit,
        "cvar": cvar_value,
        "p_DA_values": p_DA_values,
        "scenario_profit": scenario_profit,
    })
    if abs(beta) < 1e-12:
        baseline_two = frontier_two[-1]

print("Beta sweep (expected profit vs CVaR):")
for row in frontier_two:
    print(f"  beta={row['beta']:.2f} | E[profit]={row['expected_profit']:.2f} | CVaR={row['cvar']:.2f}")

# print baseline stats (beta = 0)
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
print(f"Range: €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range: {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

rows = []
for entry in frontier_two:
    beta        = entry["beta"]
    profit_dict = entry["scenario_profit"]
    p_DA_opt    = entry["p_DA_values"]
    profits     = np.array(list(profit_dict.values()))

    e_profit_all = sum(prob[omega] * profit_dict[omega] for omega in SCENARIOS)
    std          = np.std(profits)
    min_profit   = profits.min()
    cvar         = entry["cvar"]

    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]

    da_list  = []
    bal_list = []
    for omega in worst_omegas:
        da  = sum(lambda_DA.loc[omega, h] * p_DA_opt[h] for h in HOURS)
        bal = sum(
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_opt[h], 0)
            - lambda_B_down.loc[omega, h] * max(p_DA_opt[h] - wind_mw.loc[omega, h], 0)
            for h in HOURS
        )
        da_list.append(da)
        bal_list.append(bal)

    rows.append({
        "beta":            beta,
        "E[profit]":       e_profit_all,
        "std":             std,
        "min":             min_profit,
        "CVaR":            cvar,
        "E[profit|worst]": np.mean([d + b for d, b in zip(da_list, bal_list)]),
        "E[DA|worst]":     np.mean(da_list),
        "E[bal|worst]":    np.mean(bal_list),
    })

df_table = pd.DataFrame(rows)
df_table[["E[profit]", "std", "min", "CVaR",
          "E[profit|worst]", "E[DA|worst]", "E[bal|worst]"]] = \
    df_table[["E[profit]", "std", "min", "CVaR",
              "E[profit|worst]", "E[DA|worst]", "E[bal|worst]"]].round(0).astype(int)

print("\n=== Profit decomposition vs beta (two-price) ===")
print(df_table.to_string(index=False))


print("\n=== Penalized imbalance structure vs beta (two-price) ===")
print(f"{'beta':>6} {'deficit_pen%':>14} {'surplus_pen%':>14} {'avg_imbal_MW':>14}")

for entry in frontier_two:
    beta        = entry["beta"]
    profit_dict = entry["scenario_profit"]
    p_DA_opt    = entry["p_DA_values"]

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

    print(f"{beta:>6.2f} {100*n_def/total:>13.1f}% {100*n_sur/total:>13.1f}% {avg_imbal:>14.1f}")

print("\n=== Hourly penalized imbalance structure vs beta (two-price) ===")

for entry in frontier_two:
    beta        = entry["beta"]
    profit_dict = entry["scenario_profit"]
    p_DA_opt    = entry["p_DA_values"]

    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]

    print(f"\nbeta={beta:.2f}")
    print(f"{'hour':>5} {'def_pen%':>10} {'sur_pen%':>10} {'dominant':>10}")

    for h in HOURS:
        n_def = sum(
            1 for omega in worst_omegas
            if wind_mw.loc[omega, h] < p_DA_opt[h] and si.loc[omega, h] == 1
        )
        n_sur = sum(
            1 for omega in worst_omegas
            if wind_mw.loc[omega, h] > p_DA_opt[h] and si.loc[omega, h] == 0
        )
        pct_def = 100 * n_def / n_worst
        pct_sur = 100 * n_sur / n_worst

        if pct_def > pct_sur:
            dominant = "deficit"
        elif pct_sur > pct_def:
            dominant = "surplus"
        else:
            dominant = "equal"

        print(f"{h+1:>5} {pct_def:>9.1f}% {pct_sur:>9.1f}% {dominant:>10}")

from step1.plots import plot_imbalance_transition

plot_imbalance_transition(
    frontier_two, wind_mw, si, ALPHA, HOURS, SCENARIOS,
    save_path=PLOTS / "Task1.4_imbalance_transition.png",
)

# --- worst-scenario analysis (two-price, beta=0/1) ---
# identify the (1-alpha) worst scenarios by profit 
def worst_scenario_analysis(frontier_entry, label):
    profit_dict = frontier_entry["scenario_profit"]
    p_DA_opt    = frontier_entry["p_DA_values"]

    n_worst      = max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))
    worst_omegas = sorted(profit_dict, key=lambda w: profit_dict[w])[:n_worst]

    rows = []
    for omega in worst_omegas:
        for h in HOURS:
            w     = wind_mw.loc[omega, h]
            p     = p_DA_opt[h]
            imbal = w - p
            si_h  = si.loc[omega, h]

            if imbal < 0:
                penalized = (si_h == 1)
            else:
                penalized = (si_h == 0)

            rows.append({
                "omega":     omega,
                "hour":      h,
                "profit":    profit_dict[omega],
                "imbal_mw":  imbal,
                "si":        si_h,
                "penalized": penalized,
                "lda":       lambda_DA.loc[omega, h],
            })

    df = pd.DataFrame(rows)
    total   = len(df)
    n_pen   = df["penalized"].sum()
    n_def   = ((df["imbal_mw"] < 0) & df["penalized"]).sum()
    n_sur   = ((df["imbal_mw"] > 0) & df["penalized"]).sum()

    print(f"\n=== Worst-scenario analysis — two-price, {label}, alpha={ALPHA} ===")
    print(f"Scenarios analysed:  {n_worst} / {len(SCENARIOS)}")
    print(f"Total hour-slots:    {total}")
    print(f"Penalized slots:     {n_pen}  ({100*n_pen/total:.1f}%)")
    print(f"  deficit (up-reg):  {n_def}  ({100*n_def/total:.1f}%)")
    print(f"  surplus (down-reg):{n_sur}  ({100*n_sur/total:.1f}%)")
    print(f"Avg imbalance [MW]:  {df['imbal_mw'].mean():.1f}")
    print(f"Avg DA price [€/MWh]:{df['lda'].mean():.1f}")

    return df
df_worst_b0 = worst_scenario_analysis(baseline_two,      "beta=0")
df_worst_b1 = worst_scenario_analysis(frontier_two[-1],  "beta=1")

plot_profit_histogram(
    scenario_profit, prob,
    title="Profit distribution across scenarios - Two-price scheme (beta = 0)",
    save_path=PLOTS / "Task1.4_two_price_profit_distribution.png",
    color="#3fe60c",
)

frontier_two_df = pd.DataFrame(frontier_two)
plot_cvar_frontier(
    frontier_two_df,
    title="Expected Profit vs CVaR Trade-off - Two-price scheme",
    save_path=PLOTS / "Task1.4_two_price_profit_cvar_tradeoff.png",
)


plot_cvar_frontier_With_Both_Models(
    frontier_one_df, frontier_two_df, None, PLOTS / "Task1.4_both_models_profit_cvar_tradeoff.png"
) 

# profit distribution at beta=1
scenario_profit_beta1 = frontier_two[-1]["scenario_profit"]
plot_profit_histogram(
    scenario_profit_beta1, prob,
    save_path=PLOTS / "Task1.4_two_price_profit_distribution_beta1.png",
    color="#4CAF50"
) 

plot_profit_boxplot(
    frontier_one,
    color="#fa9537",
    title="Profit distribution vs. risk aversion — one-price scheme",
    save_path=PLOTS / "Task1.4_one_price_boxplot.png",
)

plot_profit_boxplot(
    frontier_two,
    color="#3fe60c",
    title="Profit distribution vs. risk aversion — two-price scheme",
    save_path=PLOTS / "Task1.4_two_price_boxplot.png",
) 

plot_profit_boxplot_comparison(
    frontier_one, frontier_two,
    save_path=PLOTS / "Task1.4_boxplot_comparison.png",
) 

worst_omegas_b0 = sorted(
    baseline_two["scenario_profit"],
    key=lambda w: baseline_two["scenario_profit"][w]
)[:max(1, math.floor(round((1 - ALPHA) * len(SCENARIOS), 6)))]

plot_hourly_offers(
    frontier_two, wind_mw, si, SCENARIOS,
    save_path=PLOTS / "Task1.4_two_price_hourly_offers_diff.png",
)