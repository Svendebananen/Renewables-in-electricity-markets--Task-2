import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent
DATA = DIR / "Data"

# load combined scenarios
scenarios = pd.read_csv(DATA / "Combined_scenarios.csv")

# parameters
P_NOM       = 500 # nominal power of the wind turbine in MW
N_HOURS     = 24  # number of hours in a day
N_SCENARIOS = scenarios['scenario_id'].nunique()
PROB        = 1 / N_SCENARIOS # probability of each scenario (assuming all scenarios are equally likely)
HOURS       = range(N_HOURS) 
SCENARIOS   = range(N_SCENARIOS)