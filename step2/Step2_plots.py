import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def per_profile_compliance(bid, sample):
    N, T = sample.shape
    compliance = np.zeros(N)
    for i in range(N):
        compliance[i] = np.mean(sample[i] >= bid)
    return compliance

def empirical_cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def plot_compliance_cdf(x_also, x_cvar, out_sample):
    also_comp = per_profile_compliance(x_also, out_sample)
    cvar_comp = per_profile_compliance(x_cvar, out_sample)

    x1, y1 = empirical_cdf(also_comp)
    x2, y2 = empirical_cdf(cvar_comp)

    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, label="ALSO-X")
    plt.plot(x2, y2, label="CVaR")
    plt.axvline(0.9, linestyle="--", color="black", label="P90 threshold")

    plt.xlabel("Per-profile compliance")
    plt.ylabel("Empirical CDF")
    plt.title("Compliance Distribution (CDF of Per-Profile Compliance)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_dir = os.path.join("Results", "Step 2 Plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "compliance_cdf.png"), dpi=300)
    plt.close()

def plot_perf_risk(epsilons, sf_in, sf_out, violation_rate_in, violation_rate_out):
    save_dir = os.path.join("Results", "Step 2 Plots")
    os.makedirs(save_dir, exist_ok=True)

    p_req = 1.0 - epsilons

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot (a): Avg Shortfall vs P Requirement ---
    axes[0].plot(p_req, sf_out, marker='o', color='tab:blue', label="Out-of-sample")
    axes[0].plot(p_req, sf_in, marker='s', color='tab:orange', linestyle='--', label="In-sample")
    axes[0].set_xlabel("P Requirement")
    axes[0].set_ylabel("Avg Shortfall (kW)")
    axes[0].grid(True)
    axes[0].legend()

    # --- Plot (b): Violation Rate vs P Requirement ---
    axes[1].plot(p_req, violation_rate_out, marker='o', color='tab:blue', label="Out-of-sample")
    axes[1].plot(p_req, violation_rate_in, marker='s', color='tab:orange', linestyle='--', label="In-sample")
    axes[1].set_xlabel("P Requirement")
    axes[1].set_ylabel("Violation Rate (%)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_2_2.png"), dpi=300)
    plt.close()

def plot_tradeoff_offer(epsilons, bids, sf_in, sf_out):
    save_dir = os.path.join("Results", "Step 2 Plots")
    os.makedirs(save_dir, exist_ok=True)

    p_req = 1.0 - epsilons

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot (a): Optimal Bid vs P Requirement ---
    axes[0].plot(p_req, bids, marker='o', color='tab:blue')
    axes[0].set_xlabel("P Requirement")
    axes[0].set_ylabel("Optimal Bid (kW)")
    axes[0].grid(True)

    # --- Plot (b): Avg Shortfall vs Optimal Bid ---
    axes[1].plot(bids, sf_out, marker='o', color='tab:blue', label="Out-of-sample")
    axes[1].plot(bids, sf_in, marker='s', color='tab:orange', linestyle='--', label="In-sample")
    axes[1].set_xlabel("Optimal Bid (kW)")
    axes[1].set_ylabel("Avg Shortfall (kW)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_2_3.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    npz_data_path = os.path.join("Results", "Step 2", "bids_results.npz")
    csv_data_path = os.path.join("Results", "Step 2", "p90_analysis_table.csv")
    
    if os.path.exists(npz_data_path) and os.path.exists(csv_data_path):
        
        # 1. CDF
        data = np.load(npz_data_path)
        plot_compliance_cdf(data["x_also"], data["x_cvar"], data["out_sample"])
        
        # 2. Figure 2.2 e 2.3
        table = pd.read_csv(csv_data_path)
        epsilons = table["epsilon"].values
        bids = table["bid"].values
        sf_in = table["sf_in"].values
        sf_out = table["sf_out"].values
        v_in = table["violation_rate_in (%)"].values
        v_out = table["violation_rate_out (%)"].values
        
        plot_perf_risk(epsilons, sf_in, sf_out, v_in, v_out)
        plot_tradeoff_offer(epsilons, bids, sf_in, sf_out)
        
        print("Grapsh generated")
    else:
        print("ERROR : Files not found.")