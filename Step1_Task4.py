"""
Task 1.4) Risk-Averse Offering Strategy
"""

import pandas as pd
import numpy as np
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS
)
from step1.models import (
    compute_balancing_prices_one, compute_balancing_prices_two,
    solve_one_price, solve_two_price,
)
from step1.plots import plot_cvar_frontier_With_Both_Models, plot_profit_histogram, plot_cvar_frontier

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