# Renewables in electricity markets: Assignment 2

* Step 1:
- Electricity prices have been extracted from https://data.open-power-system-data.org/time_series/, since
  Entso allows the download of single days data. This website is more conveniente and we 
  can sample the scenarios randomly from the whole year multiple times.

- K-means clustering was used instead of random sampling to ensure that the 20 representative profiles for
  wind and price cover the full annual variability proportionally, reducing the risk of underrepresenting specific seasonal patterns. (I'll ask the TAs if it is ok, otherwise we can come back to the previous version in which I implemented only the random sampling of price and wind scenarios from the 2019 dataset)      

- Results: as in one of Jalal's papers, we have tha the one-price scheme is leading to an all-or-nothing strategy.
  The wind farm is bidding 100% of its full capacity in the DA if the sum of all scenarios Prob * (lambda_DA - lambda_B) is positive, otherwise it bids 0. E.g. if we expect that the balancing price is > than the DA price, it's better for us leave th whole production for the balancing market, in which we are going to be remunerated with a factor that is 1.25 higher than in the DA.

* Step 2
Data
-Generate_load_scenarios creates random 300 profiles, reproducibility attained by using seed, the report uses seed = 60
-Step2_solvers: Gurobi solvers for alsox and cvar that can be imported in Step2.py

Step 2 (Task 2. to 2.3): Executing this file generates profiles and use 100 profiles to determine the optimal FCR-D UP reserve bid (in kW) satisfying Energinet’s P90 requirement by using both ALSO-X and CVaR. Verify of the P90 requiremnte by using the 200 out of sample loads, evalutaing compliance rate, avg. shortfall and 95th percentile shortfall. Varying the P90 requirement from 80% to 100% requirement and analyzing the effect on 
optimal reserve bid (in-sample) and the expected reserve shortfall (out-of-sample) for ALSO-X.
Step2_plots (Plotting): Imports the results from Step 2 to for various plots useed for visualizing the results: 1.) Compliance vs. CDF for both methods 2.) Influence of P90 requirement on ALSOX
