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

    m_count, omega = arr.shape
    return arr, m_count, omega


def solve_also_x_gurobi(
    profiles: np.ndarray,
    epsilon: float,
    M: float = 1e5,
    output_flag: int = 0,
) -> float:
    """Solve ALSO-X MILP and return the optimal upward reserve bid."""
    arr, m_count, omega = _validate_inputs(profiles, epsilon)

    model = gp.Model("also_x")
    model.setParam("OutputFlag", output_flag)

    c_up = model.addVar(lb=0.0, name="c_up")
    y = model.addVars(m_count, omega, vtype=GRB.BINARY, name="y")

    model.setObjective(c_up, GRB.MAXIMIZE)

    model.addConstrs(
        (
            c_up - float(arr[mp, w]) <= M * y[mp, w]
            for mp in range(m_count)
            for w in range(omega)
        ),
        name="link",
    )

    model.addConstr(
        gp.quicksum(y[mp, w] for mp in range(m_count) for w in range(omega))
        <= epsilon * m_count * omega,
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
    """Solve CVaR LP and return the optimal upward reserve bid."""
    
    arr, m_count, omega = _validate_inputs(profiles, epsilon)
    total_samples = m_count * omega

    model = gp.Model("cvar")
    model.setParam("OutputFlag", output_flag)

    # Variables
    c_up = model.addVar(lb=0.0, name="c_up")
    beta = model.addVar(lb=-GRB.INFINITY, name="beta")  # VaR
    zeta = model.addVars(m_count, omega, lb=0.0, name="zeta")

    # Objective
    model.setObjective(c_up, GRB.MAXIMIZE)

    model.addConstrs(
        (
            zeta[m, w] >= c_up - float(arr[m, w]) - beta
            for m in range(m_count)
            for w in range(omega)
        ),
        name="shortfall",
    )

    
    model.addConstr(
        beta + (1.0 / (epsilon * total_samples)) *
        gp.quicksum(zeta[m, w] for m in range(m_count) for w in range(omega))
        <= 0,
        name="cvar_constraint",
    )

    model.optimize()

    
    if model.status != GRB.OPTIMAL:
        raise RuntimeError("CVar solver failed")

    return float(c_up.X)

    r



