import numpy as np
import time
import os
from Data.Step2_solvers import solve_also_x_gurobi, solve_cvar_gurobi
from Data.Generate_load_scenarios import generate_load_scenarios

############################################################################
### Main execution: generate data, solve models, evaluate & save results ###
############################################################################
def main():
    start_time = time.time()

    # 1. Generate and split profiles

    profiles   = generate_load_scenarios(num_scenarios=300, num_steps=60, seed=60)
    rng        = np.random.default_rng(0)
    perm       = rng.permutation(profiles.shape[0])
    in_sample  = profiles[perm[:100], :]
    out_sample = profiles[perm[100:], :]

    # 2. Solve ALSO-X and CVaR for ε=0.1
    x_also = solve_also_x_gurobi(in_sample, epsilon=0.1)
    elapsed_also = time.time() - start_time
    start_time = time.time()
    x_cvar = solve_cvar_gurobi(in_sample,   epsilon=0.1)
    elapsed_cvar = time.time() - start_time
    

    # 3. Print optimal bids
    print("Optimal bids:")
    print(f"  ALSO-X: {x_also:.2f} kW")
    print(f"  CVaR:   {x_cvar:.2f} kW\n")
    # 4. Evaluate in-sample violations
    N, T = in_sample.shape
    total_minutes = N * T
    print("In-sample performance:")
    for name, bid, sample in [("ALSO-X", x_also, in_sample),
                         ("CVaR",   x_cvar, in_sample)]:
        
        violations = np.count_nonzero(sample < bid)
        rate       = violations / total_minutes * 100
        shortfall  = np.maximum(0, bid - sample).mean()
        shortfall_95 = np.percentile(np.maximum(0, bid - sample).flatten(), 95)

        runtime = elapsed_also if name == "ALSO-X" else elapsed_cvar

        print(f"  {name:6} → {violations} violations ,"
            f"violation rate{rate:.2f}%," 
            f"Runtime: {runtime:.2f} seconds")
    print()

    # 5. Evaluate out-of-sample violations
    N, T = out_sample.shape
    total_minutes = N * T
    print("Out-of-sample performance:")
    for name, bid, sample in [("ALSO-X", x_also, out_sample),
                              ("CVaR",   x_cvar,   out_sample)]:
        violations = np.count_nonzero(sample < bid)
        rate       = violations / total_minutes * 100
        shortfall  = np.maximum(0, bid - sample).mean()
        shortfall_95 = np.percentile(np.maximum(0, bid - sample).flatten(), 95) # 95th percentile of shortfall
        print(f"  {name:6} → {violations} violations "
              f"({rate:.2f}%), avg shortfall {shortfall:.2f} kW"
              f", 95th percentile shortfall {shortfall_95:.2f} kW"
              )
    print()

    # 6. Save results 
    
    np.savez(
    os.path.join("data", "bids_results.npz"),
    x_also=x_also,
    x_cvar=x_cvar,
    out_sample=out_sample
)

    

if __name__ == "__main__":
    main()








