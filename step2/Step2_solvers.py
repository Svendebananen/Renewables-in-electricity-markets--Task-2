"""
Reusable Gurobi solver functions for Step 2 tasks.

Import in task scripts with:
    from Step2_solvers import solve_also_x_gurobi, solve_cvar_gurobi

"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _validate_inputs(profiles: np.ndarray, epsilon: float) -> tuple[np.ndarray, int, int]:
    """Validate common inputs and return normalized data and shape."""
    arr = np.asarray(profiles, dtype=float)

    if arr.ndim != 2:
        raise ValueError("profiles must be a 2D array-like object")
    if arr.size == 0:
        raise ValueError("profiles must not be empty")
    if not (0 <= epsilon <= 1):
        raise ValueError("epsilon must satisfy 0 < epsilon <= 1")

    n_scenarios, n_minutes = arr.shape
    return arr, n_scenarios, n_minutes


def solve_also_x_gurobi(
    profiles: np.ndarray,
    epsilon: float,
    M: float = 1e5,
    output_flag: int = 0,
) -> float:
    """Solve ALSO-X MILP and return the optimal upward reserve bid."""
    arr, n_scenarios, n_minutes = _validate_inputs(profiles, epsilon)

    model = gp.Model("also_x")
    model.setParam("OutputFlag", output_flag)

    c_up = model.addVar(lb=0.0, name="c_up")
    y = model.addVars(n_scenarios, n_minutes, vtype=GRB.BINARY, name="y")

    model.setObjective(c_up, GRB.MAXIMIZE)

    model.addConstrs(
        (
            c_up - float(arr[w, m]) <= M * y[w, m]
            for w in range(n_scenarios)
            for m in range(n_minutes)
        ),
        name="link",
    )

    model.addConstr(
        gp.quicksum(y[w, m] for w in range(n_scenarios) for m in range(n_minutes))
        <= epsilon * n_scenarios * n_minutes,
        name="budget",
    )

    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError("ALSO-X solver failed")

    return float(c_up.X)


def solve_cvar_gurobi(
    profiles: np.ndarray,
    epsilon: float,
    output_flag: int = 0,
) -> float:
    """Solve CVaR LP and return the optimal upward reserve bid.

    Enforces: CVaR_{1-epsilon}(c_up - load) <= 0
    LP reformulation: beta + (1 / (epsilon * N)) * sum(s) <= 0
                      s[w,m] >= (c_up - load[w,m]) - beta
                      s[w,m] >= 0
    """
    arr, n_scenarios, n_minutes = _validate_inputs(profiles, epsilon)
    N = n_scenarios * n_minutes

    model = gp.Model("cvar")
    model.setParam("OutputFlag", output_flag)

    c_up = model.addVar(lb=0.0, name="c_up")
    beta = model.addVar(lb=-GRB.INFINITY, name="beta")          # VaR level
    s = model.addVars(n_scenarios, n_minutes, lb=0.0, name="s") # excess shortfall above VaR

    model.setObjective(c_up, GRB.MAXIMIZE)

    model.addConstrs(
        (s[w, m] >= c_up - float(arr[w, m]) - beta
         for w in range(n_scenarios)
         for m in range(n_minutes)),
        name="excess_shortfall",
    )

    model.addConstr(
        beta + (1.0 / (epsilon * N)) *
        gp.quicksum(s[w, m] for w in range(n_scenarios) for m in range(n_minutes))
        <= 0.0,
        name="CVaR_bound",
    )

    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError("CVaR solver failed")

    return float(c_up.X)