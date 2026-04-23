"""
step1/models.py
Solver functions for the one-price and two-price balancing market models,
plus helpers that compute the scenario balancing prices.

Both solvers include optional CVaR regularisation controlled by `beta`
(0 = pure expected-profit maximisation, 1 = pure CVaR maximisation).

Return value for both solvers:
    p_DA_values     : dict  {h: float}
    scenario_profit : dict  {omega: float}
    cvar_value      : float  (CVaR at the chosen alpha level)
"""
import gurobipy as gp
from gurobipy import GRB

DEFICIT_MULTIPLIER = 1.25
SURPLUS_MULTIPLIER = 0.85


# ---------------------------------------------------------------------------
# Balancing-price helpers
# ---------------------------------------------------------------------------

def compute_balancing_prices_one(lambda_DA, si):
    """
    One-price scheme: a single balancing price per scenario/hour.
    Returns lambda_B (same shape as lambda_DA).
    """
    lambda_B = lambda_DA.copy()
    for omega in lambda_DA.index:
        for h in lambda_DA.columns:
            if si.loc[omega, h] == 1:                                    # deficit
                lambda_B.loc[omega, h] = DEFICIT_MULTIPLIER * lambda_DA.loc[omega, h]
            else:                                                         # surplus
                lambda_B.loc[omega, h] = SURPLUS_MULTIPLIER * lambda_DA.loc[omega, h]
    return lambda_B


def compute_balancing_prices_two(lambda_DA, si):
    """
    Two-price scheme: separate up/down balancing prices per scenario/hour.
    Returns (lambda_B_up, lambda_B_down), each same shape as lambda_DA.
    """
    lambda_B_up   = lambda_DA.copy()
    lambda_B_down = lambda_DA.copy()
    for omega in lambda_DA.index:
        for h in lambda_DA.columns:
            if si.loc[omega, h] == 1:                                    # upward need
                lambda_B_up.loc[omega, h]   = lambda_DA.loc[omega, h]           # beneficial
                lambda_B_down.loc[omega, h] = DEFICIT_MULTIPLIER * lambda_DA.loc[omega, h]  # harmful
            else:                                                         # downward need
                lambda_B_up.loc[omega, h]   = SURPLUS_MULTIPLIER * lambda_DA.loc[omega, h]  # harmful
                lambda_B_down.loc[omega, h] = lambda_DA.loc[omega, h]           # beneficial
    return lambda_B_up, lambda_B_down


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_one_price(
    scenarios, prob, wind_mw, lambda_DA, lambda_B,
    *, beta=0, alpha=0.9, verbose=False
):
    """
    Solve the one-price stochastic offering model (with optional CVaR term).

    Parameters
    ----------
    scenarios : list of scenario IDs to include
    prob      : Series of scenario probabilities (indexed by scenario ID)
    wind_mw   : DataFrame of wind power realisations (scenarios × hours)
    lambda_DA : DataFrame of day-ahead prices (scenarios × hours)
    lambda_B  : DataFrame of balancing prices (scenarios × hours)
    beta      : float in [0, 1] – weight on CVaR term (0 = pure E[profit])
    alpha     : float in (0, 1) – CVaR confidence level
    verbose   : bool – if True, show Gurobi solver output

    Returns
    -------
    p_DA_values     : dict {h: float}
    scenario_profit : dict {omega: float}
    cvar_value      : float
    """
    hours   = list(lambda_DA.columns)
    p_nom   = wind_mw.max().max()   # upper bound – use actual P_NOM if preferred

    model = gp.Model("one_price")
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Decision variables
    p_DA = model.addVars(hours, lb=0, ub=p_nom, name="p_DA")
    Aux  = model.addVars(scenarios, lb=0, name="Aux")      # CVaR shortfall
    Etha = model.addVar(lb=-GRB.INFINITY, name="Etha")     # VaR auxiliary

    # Scenario-profit expressions
    scenario_profit_expr = {
        omega: gp.quicksum(
            lambda_DA.loc[omega, h] * p_DA[h] +
            lambda_B.loc[omega, h]  * (wind_mw.loc[omega, h] - p_DA[h])
            for h in hours
        )
        for omega in scenarios
    }

    # Expected profit & CVaR expressions
    expected_profit_expr = gp.quicksum(
        prob[omega] * scenario_profit_expr[omega]
        for omega in scenarios
    )
    cvar_expr = Etha - (1 / (1 - alpha)) * gp.quicksum(
        prob[omega] * Aux[omega] for omega in scenarios
    )

    model.setObjective(
        (1 - beta) * expected_profit_expr + beta * cvar_expr,
        GRB.MAXIMIZE
    )

    # CVaR shortfall constraints: Aux[omega] >= Etha - profit[omega]
    model.addConstrs(
        (Aux[omega] >= Etha - scenario_profit_expr[omega] for omega in scenarios),
        name="cvar_shortfall"
    )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution (status={model.status})")

    p_DA_values = {h: p_DA[h].X for h in hours}
    scenario_profit = {
        omega: sum(
            lambda_DA.loc[omega, h] * p_DA_values[h] +
            lambda_B.loc[omega, h]  * (wind_mw.loc[omega, h] - p_DA_values[h])
            for h in hours
        )
        for omega in scenarios
    }
    cvar_value = Etha.X - (1 / (1 - alpha)) * sum(
        prob[omega] * Aux[omega].X for omega in scenarios
    )

    return p_DA_values, scenario_profit, cvar_value


def solve_two_price(
    scenarios, prob, wind_mw, lambda_DA, lambda_B_up, lambda_B_down,
    *, beta=0, alpha=0.9, verbose=False
):
    """
    Solve the two-price stochastic offering model (with optional CVaR term).

    Parameters
    ----------
    scenarios     : list of scenario IDs to include
    prob          : Series of scenario probabilities (indexed by scenario ID)
    wind_mw       : DataFrame of wind power realisations (scenarios × hours)
    lambda_DA     : DataFrame of day-ahead prices (scenarios × hours)
    lambda_B_up   : DataFrame of upward balancing prices (scenarios × hours)
    lambda_B_down : DataFrame of downward balancing prices (scenarios × hours)
    beta          : float in [0, 1] – weight on CVaR term
    alpha         : float in (0, 1) – CVaR confidence level
    verbose       : bool – if True, show Gurobi solver output

    Returns
    -------
    p_DA_values     : dict {h: float}
    scenario_profit : dict {omega: float}
    cvar_value      : float
    """
    hours = list(lambda_DA.columns)
    p_nom = wind_mw.max().max()

    model = gp.Model("two_price")
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Decision variables
    p_DA       = model.addVars(hours,             lb=0, ub=p_nom, name="p_DA")
    delta_up   = model.addVars(scenarios, hours,  lb=0, ub=p_nom, name="delta_up")
    delta_down = model.addVars(scenarios, hours,  lb=0, ub=p_nom, name="delta_down")
    Aux        = model.addVars(scenarios,         lb=0,            name="Aux")
    Etha       = model.addVar(lb=-GRB.INFINITY,                    name="Etha")

    # Balance constraints: delta_up - delta_down = wind - p_DA
    for omega in scenarios:
        for h in hours:
            model.addConstr(
                delta_up[omega, h] - delta_down[omega, h]
                == wind_mw.loc[omega, h] - p_DA[h]
            )

    # Scenario-profit expressions
    scenario_profit_expr = {
        omega: gp.quicksum(
            lambda_DA.loc[omega, h]     * p_DA[h] +
            lambda_B_up.loc[omega, h]   * delta_up[omega, h] -
            lambda_B_down.loc[omega, h] * delta_down[omega, h]
            for h in hours
        )
        for omega in scenarios
    }

    # Expected profit & CVaR expressions
    expected_profit_expr = gp.quicksum(
        prob[omega] * scenario_profit_expr[omega]
        for omega in scenarios
    )
    cvar_expr = Etha - (1 / (1 - alpha)) * gp.quicksum(
        prob[omega] * Aux[omega] for omega in scenarios
    )

    model.setObjective(
        (1 - beta) * expected_profit_expr + beta * cvar_expr,
        GRB.MAXIMIZE
    )

    # CVaR shortfall constraints
    model.addConstrs(
        (Aux[omega] >= Etha - scenario_profit_expr[omega] for omega in scenarios),
        name="cvar_shortfall"
    )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution (status={model.status})")

    p_DA_values = {h: p_DA[h].X for h in hours}
    delta_up_vals   = {(omega, h): delta_up[omega, h].X   for omega in scenarios for h in hours}
    delta_down_vals = {(omega, h): delta_down[omega, h].X for omega in scenarios for h in hours}

    scenario_profit = {
        omega: sum(
            lambda_DA.loc[omega, h]     * p_DA_values[h] +
            lambda_B_up.loc[omega, h]   * delta_up_vals[omega, h] -
            lambda_B_down.loc[omega, h] * delta_down_vals[omega, h]
            for h in hours
        )
        for omega in scenarios
    }
    cvar_value = Etha.X - (1 / (1 - alpha)) * sum(
        prob[omega] * Aux[omega].X for omega in scenarios
    )

    return p_DA_values, scenario_profit, cvar_value
