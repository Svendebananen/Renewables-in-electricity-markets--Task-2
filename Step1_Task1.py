import numpy as np
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS
)
from step1.models import compute_balancing_prices_one, solve_one_price
from step1.plots import plot_profit_histogram

# compute balancing prices
lambda_B = compute_balancing_prices_one(lambda_DA, si)

# solve one-price model (beta=0: pure expected-profit maximisation)
p_DA_values, scenario_profit, _ = solve_one_price(
    SCENARIOS, prob, wind_mw, lambda_DA, lambda_B, verbose=True
)

# hourly expected profit
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

total_profit = sum(hourly_profit.values())

print(f"Total expected profit: {total_profit:.2f} €")
print("Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")

# statistical analysis of scenarios
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

# plot profit distribution
plot_profit_histogram(
    scenario_profit, prob,
    title="Profit distribution across scenarios - One-price scheme",
    save_path=PLOTS / "Task1.1_profit_distribution.png",
    color="#fa9537"
)