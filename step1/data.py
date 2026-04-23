"""
step1/data.py
Loads Combined_scenarios.csv and exposes all shared data objects used
across Step 1 tasks.
"""
import pandas as pd
from pathlib import Path

# Paths
DIR   = Path(__file__).parent.parent
DATA  = DIR / "Data"
PLOTS = DIR / "Step 1 Plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Load raw scenarios
scenarios = pd.read_csv(DATA / "Combined_scenarios.csv")

# Pivot to matrix format (scenarios × hours)
wind_mw   = scenarios.pivot(index='scenario_id', columns='hour', values='wind_mw')
lambda_DA = scenarios.pivot(index='scenario_id', columns='hour', values='da_price')
si        = scenarios.pivot(index='scenario_id', columns='hour', values='si')
prob      = (
    scenarios
    .drop_duplicates('scenario_id')
    .set_index('scenario_id')['prob']
)

# Constants
P_NOM     = 500          # nominal power of the wind turbine (MW)
N_HOURS   = 24
HOURS     = range(N_HOURS)
SCENARIOS = list(wind_mw.index)
