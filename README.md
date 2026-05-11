# 46755_REM_Assignment2

Stochastic optimization in electricity markets. Developed as an assignment for **46755 Renewables in Electricity Markets** at the Technical University of Denmark (DTU) - MSc in Sustainable Energy Systems.

---

## Overview

This repository automates key steps in market participation strategies for renewable energy assets and flexible loads under uncertainty. Leveraging data analysis and advanced optimization, the package:

- Generates stochastic wind, price, and system imbalance scenarios using K-means clustering.
- Evaluates optimal Day-Ahead offering strategies under One-Price and Two-Price balancing schemes.
- Performs 8-fold cross-validation to assess the out-of-sample performance of market bids.
- Traces the efficient frontier of Expected Profit vs. Conditional Value-at-Risk (CVaR).
- Generates high-resolution (minute-level) stochastic load profiles for flexible consumption.
- Solves joint chance-constrained optimization problems for the FCR-D UP ancillary market using Exact (ALSO-X MILP) and Convex Approximation (CVaR LP) methods.
- Visualizes empirical compliance distributions and sensitivity analysis metrics.

---

## Repository Structure & Module Description

The codebase is divided into scenario generation, market optimization (Step 1), and ancillary services (Step 2).

### Data & Scenario Generation

| File | Description |
| --- | --- |
| `Step1_scenario_generation.py` | Uses `sklearn.cluster.KMeans` to reduce historical data into 20 representative wind and 20 price profiles. Applies a binomial distribution for system imbalances to output 1600 combined scenarios. |
| `Generate_load_scenarios.py` | Simulates high-resolution minute-by-minute stochastic load variations. Ensures physical constraints (max 35 kW change/min, bounded between 220-600 kW). |

### Step 1: Day-Ahead & Balancing Markets

| File | Description |
| --- | --- |
| `Step1_Task1AndTask2.py` | Computes the optimal Day-Ahead wind offer ($p^{DA}$) maximizing expected profit under symmetric (One-Price) and asymmetric (Two-Price) balancing penalties. |
| `Step1_Task3.py` | Implements an 8-fold cross-validation pipeline. Splits the 1600 scenarios into training (in-sample) and test (out-of-sample) sets to evaluate the robustness of the Two-Price bidding strategy. |
| `Step1_Task4.py` | Incorporates risk-aversion. Iterates over multiple $\beta$ values to trace the Efficient Frontier, balancing Expected Profit against Conditional Value-at-Risk (CVaR). |

### Step 2: Ancillary Service Markets (FCR-D UP)

| File / Component | Description |
| --- | --- |
| `Step2_solvers.py` | Core Gurobi optimization module containing the mathematical formulations. |
| ↳ `solve_also_x_gurobi()` | Implements the **ALSO-X** method as a Mixed-Integer Linear Program (MILP). Uses binary variables $y_{m,\omega}$ and a Big-M formulation to strictly bound the number of reserve shortfalls. |
| ↳ `solve_cvar_gurobi()` | Implements the **CVaR** convex approximation as a Linear Program (LP). Uses auxiliary variables $s_{m,\omega}$ to limit the expected magnitude of tail-end violations. |
| `Step2.py` | Main execution script. Splits 300 load profiles, calls the solvers to determine the optimal $c^{\uparrow}$ bid, and evaluates in-sample and out-of-sample violation rates and average shortfalls. |
| `Step2_plots.py` | Analytics and visualization module. Computes empirical CDFs of per-profile compliance and generates a 4-panel sensitivity analysis across different $P_{90}$ reliability requirements. |

---

## Git Workflow and Collaboration

We used a **feature branch workflow** to manage the mathematical models and the data pipelines:

* All work is done on feature branches, never directly on `main`.
* Branch naming convention: `feature/<description>` (e.g., `feature/cvar-optimization`).
* Pull requests (PRs) are opened on GitHub for code review.

Work was collaboratively divided to play to our strengths

---

## Running the Project

To execute the Day-Ahead market analysis:

```bash
python Step1_Task1AndTask2.py
python Step1_Task3.py
python Step1_Task4.py

```

To execute the Ancillary Services optimization and generate plots:

```bash
python Step2.py
python step2\Step2_plots.py  

