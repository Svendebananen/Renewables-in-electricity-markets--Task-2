# Renewables in electricity markets Assignment 2

Step 1:
- Electricity prices have been extracted from https://data.open-power-system-data.org/time_series/, since
  Entso allows the download of single days data. This website is more conveniente and we 
  can sample the scenarios randomly from the whole year multiple times.

- K-means clustering was used instead of random sampling to ensure that the 20 representative profiles for wind and price cover the full annual variability proportionally, reducing the risk of underrepresenting specific seasonal patterns. (I'll ask the TAs if it is ok, otherwise we can come back to the previous version in which I implemented only the random sampling of price and wind scenarios from the 2019 dataset)

Step 2
- Maybe the generate_load_scenarios.py should be deleted. So we just have the csv file. 
    -> i think we should keep it anyways in a separate folder to show how we obtained them
