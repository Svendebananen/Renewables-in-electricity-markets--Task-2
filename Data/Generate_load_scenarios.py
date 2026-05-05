"""
Generate random load scenarios for a Participation in Ancillary Service Markets model
for a single hour.

Constraints:
  - Load range      : 220 kW to 600 kW
  - Max change/min  : 35 kW
  - Duration        : 60 minutes (12:00 – 13:00, inclusive)

Output: load_scenarios.csv
  Columns: Scenario, Time, Load
"""

import random
import csv
import argparse
import numpy as np
from datetime import datetime, timedelta

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_SCENARIOS = 300
LOAD_MIN = 220          # kW
LOAD_MAX = 600          # kW
MAX_DELTA = 35          # kW per minute
START_TIME = datetime(2000, 1, 1, 12, 0)   # 12:00 (date is arbitrary)
NUM_STEPS = 60          # minutes (produces 61 time points: 12:00 … 12:59)
OUTPUT_FILE = "load_scenarios.csv"
DEFAULT_SEED = 60


def generate_scenario(rng: random.Random) -> list[tuple[str, int]]:
    """
    Generate a single load scenario that satisfies:
      - Load stays within [LOAD_MIN, LOAD_MAX]
      - Consecutive values differ by at most MAX_DELTA

    Returns a list of (time_str, load) tuples for each minute, where
    time_str follows the '%H:%M' format (e.g. '12:00', '12:01', …, '13:00').
    """
    # Pick a random starting load within the allowed range
    load = rng.randint(LOAD_MIN, LOAD_MAX)
    scenario: list[tuple[str, int]] = []

    for step in range(NUM_STEPS):          # 0 … 59
        timestamp = START_TIME + timedelta(minutes=step)
        time_str = timestamp.strftime("%H:%M")
        scenario.append((time_str, load))

        if step < NUM_STEPS:
            # Determine the reachable range for the next minute
            lo = max(LOAD_MIN, load - MAX_DELTA)
            hi = min(LOAD_MAX, load + MAX_DELTA)
            load = rng.randint(lo, hi)

    return scenario


def generate_load_scenarios(
    num_scenarios: int = NUM_SCENARIOS,
    num_steps: int = NUM_STEPS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Generate a scenario matrix with shape (num_scenarios, num_steps + 1)."""
    rng = random.Random(seed)
    profiles = np.zeros((num_scenarios, num_steps ), dtype=float)

    for scenario_idx in range(num_scenarios):
        scenario = generate_scenario(rng)
        profiles[scenario_idx, :] = [load for _, load in scenario]

    return profiles


def main(seed: int = DEFAULT_SEED) -> None:
    """Generate NUM_SCENARIOS scenarios and write them to OUTPUT_FILE."""
    rng = random.Random(seed)

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "Time", "Load"])

        for scenario_num in range(1, NUM_SCENARIOS + 1):
            scenario = generate_scenario(rng)
            for time_str, load in scenario:
                writer.writerow([scenario_num, time_str, load])

    print(f"Generated {NUM_SCENARIOS} scenarios with seed {seed} → {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random load scenarios.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible scenarios (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()
    main(seed=args.seed)
