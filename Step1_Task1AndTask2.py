"""
Tasks 1.1 and 1.2:
Task 1.1: Offering Strategy Under a One-Price Balancing Scheme
Task 1.2: Offering Strategy Under a Two-Price Balancing Scheme
"""

import numpy as np
from step1.data import (
    wind_mw, lambda_DA, si, prob, SCENARIOS, HOURS, PLOTS
)

from step1.models import compute_balancing_prices_one, solve_one_price, compute_balancing_prices_two, solve_two_price
from step1.plots import plot_Expected_DA_And_Balancing_Values, plot_Mean_Wind_And_DA_Offers, plot_profit_histogram, plot_hourly_offers, plot_hourly_offers_and_prices, plot_profit_histogram_comparison, plot_offers_and_prices_two

# -------- Task 1 --------
print("\n------ Task 1: One-price scheme ------")
# compute balancing prices
lambda_B = compute_balancing_prices_one(lambda_DA, si)

# solve one-price model (beta=0: pure expected-profit maximisation)
p_DA_One_Price_values, scenario_profit_one, _, Day_Ahead_Revenue_One_Price, Balancing_Revenue_One_Price = solve_one_price(
    SCENARIOS, prob, wind_mw, lambda_DA, lambda_B, verbose=False
)

# hourly expected profit
hourly_profit = {
    h: sum(
        prob[omega] * (
            lambda_DA.loc[omega, h] * p_DA_One_Price_values[h] +
            lambda_B.loc[omega, h]  * (wind_mw.loc[omega, h] - p_DA_One_Price_values[h])
        )
        for omega in SCENARIOS
    )
    for h in HOURS
}

total_profit = sum(hourly_profit.values())

print(f"Total expected profit: {total_profit:.2f} €")
print("Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_One_Price_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")


# expected day ahead value and expected balancing value for each hour to determine the decision of how much to offer in the day-ahead market
expected_DA_value = {}
expected_balancing_value = {}
print("\nExpected day-ahead value and expected balancing value for each hour:")
for h in HOURS:
    expected_DA_value[h] = sum(prob[omega] * lambda_DA.loc[omega, h] for omega in SCENARIOS)
    expected_balancing_value[h] = sum(prob[omega] * lambda_B.loc[omega, h] for omega in SCENARIOS)
    print(f"  Hour {h:2d}: Expected DA value = {expected_DA_value[h]:.2f} €/MWh, Expected balancing value = {expected_balancing_value[h]:.2f} €/MWh, difference = {expected_DA_value[h] - expected_balancing_value[h]:.2f} €/MWh")

# statistical analysis of scenarios
profits_array = np.array(list(scenario_profit_one.values()))
min_val   = profits_array.min()
max_val   = profits_array.max()
range_val = max_val - min_val

print()
print(f"Range: €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Day-Ahead Revenue:  €{Day_Ahead_Revenue_One_Price:.2f}")
print(f"Balancing Revenue:  €{Balancing_Revenue_One_Price:.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range: {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

plot_hourly_offers(
    p_DA_One_Price_values, HOURS,
    title="Optimal day-ahead offers - One-price scheme",
    save_path=PLOTS / "Task1.1_hourly_offers.png",
    color="#fa9537"
)
# single plot with offers and prices
plot_hourly_offers_and_prices(
    p_DA_One_Price_values, expected_balancing_value, expected_DA_value, HOURS,
    #title="Optimal day-ahead offers and expected prices - One-price scheme",
    save_path=PLOTS / "Task1.1_hourly_offers_and_prices.png"
)

# plot profit distribution
plot_profit_histogram(
    scenario_profit_one, prob,
    #title="Profit distribution across scenarios - One-price scheme",
    save_path=PLOTS / "Task1.1_profit_distribution.png",
    color="#fa9537"
)

# plot expected day ahead value, expected balancing value for each hour to determine the decision of how much to offer in the day-ahead market
plot_Expected_DA_And_Balancing_Values(
    expected_balancing_value, expected_DA_value, HOURS,
    save_path=PLOTS / "Task1.1_expected_da_and_balancing_values.png",
    color="#fa9537"
)


# -------- Task 2 --------
print()
print("------ Task 2: Two-price scheme ------")

# compute two-price balancing prices
lambda_B_up, lambda_B_down = compute_balancing_prices_two(lambda_DA, si)

# solve two-price model (beta=0: pure expected-profit maximisation)
p_DA_Two_Price_values, scenario_profit_two, _, Day_Ahead_Revenue_Two_Price, Balancing_Revenue_Two_Price = solve_two_price(
    SCENARIOS, prob, wind_mw, lambda_DA, lambda_B_up, lambda_B_down, verbose=False
)

# hourly expected profit
hourly_profit = {
    h: sum(
        prob[omega] * (
            lambda_DA.loc[omega, h]     * p_DA_Two_Price_values[h] +
            lambda_B_up.loc[omega, h]   * max(wind_mw.loc[omega, h] - p_DA_Two_Price_values[h], 0) -
            lambda_B_down.loc[omega, h] * max(p_DA_Two_Price_values[h] - wind_mw.loc[omega, h], 0)
        )
        for omega in SCENARIOS
    )
    for h in HOURS
}

total_profit = sum(hourly_profit.values())

print(f"Total expected profit: {total_profit:.2f} €")
print("Hourly offers and profits:")
for h in HOURS:
    print(f"  Hour {h:2d}: p_DA = {p_DA_Two_Price_values[h]:.2f} MW, profit = {hourly_profit[h]:.2f}")


# expected day ahead value and expected balancing value for each hour to determine the decision of how much to offer in the day-ahead market

print("\nExpected day-ahead value and expected balancing value for each hour:")
expected_Two_price_DA_value             = {}
expected_Two_Price_balancing_up_value   = {}
expected_Two_Price_balancing_down_value = {}

for h in HOURS:
    expected_Two_price_DA_value[h]             = sum(prob[omega] * lambda_DA.loc[omega, h] for omega in SCENARIOS)
    expected_Two_Price_balancing_up_value[h]   = sum(prob[omega] * lambda_B_up.loc[omega, h] for omega in SCENARIOS)
    expected_Two_Price_balancing_down_value[h] = sum(prob[omega] * lambda_B_down.loc[omega, h] for omega in SCENARIOS)
    print(f"  Hour {h:2d}: Expected DA value = {expected_Two_price_DA_value[h]:.2f} €/MWh, "
          f"Expected balancing up value = {expected_Two_Price_balancing_up_value[h]:.2f} €/MWh, "
          f"Expected balancing down value = {expected_Two_Price_balancing_down_value[h]:.2f} €/MWh")


# statistical analysis of scenarios
profits_array = np.array(list(scenario_profit_two.values()))
min_val   = profits_array.min()
max_val   = profits_array.max()
range_val = max_val - min_val

print()
print(f"Range: €{range_val:,.2f}")
print(f"Expected profit:    €{np.average(profits_array, weights=list(prob.loc[SCENARIOS])):.2f}")
print(f"Day-Ahead Revenue:  €{Day_Ahead_Revenue_Two_Price:.2f}")
print(f"Balancing Revenue:  €{Balancing_Revenue_Two_Price:.2f}")
print(f"Standard deviation: €{np.std(profits_array):.2f}")
print(f"Std Dev / Range: {np.std(profits_array) / range_val:.1%}")
print(f"Minimum profit:     €{min_val:,.2f}")
print(f"Maximum profit:     €{max_val:,.2f}")
print(f"Median profit:      €{np.median(profits_array):.2f}")

# plot profit distribution
plot_profit_histogram(
    scenario_profit_two, prob,
    #title="Profit distribution across scenarios - Two-price scheme",
    save_path=PLOTS / "Task1.2_profit_distribution.png",
    color="#3A7D8C"
)

# plot mean wind generation and day-ahead price across hours
plot_Mean_Wind_And_DA_Offers(
    wind_mw, p_DA_One_Price_values, p_DA_Two_Price_values, HOURS,
    save_path=PLOTS / "Task1.2_mean_wind_and_da_offers.png",
    color="#fa9537"
)

# plot profit distribution comparison
plot_profit_histogram_comparison(
    scenario_profit_one, scenario_profit_two, prob,
    save_path=PLOTS / "Task1.2_profit_distribution_comparison.png"
) 


plot_offers_and_prices_two(
    wind_mw, p_DA_One_Price_values, p_DA_Two_Price_values,
    expected_Two_price_DA_value, 
    expected_Two_Price_balancing_up_value, 
    expected_Two_Price_balancing_down_value,
    HOURS, save_path=PLOTS / "Task1.2_offers_and_prices.png"
)