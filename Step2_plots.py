import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def per_profile_compliance(bid, sample):
    """
    Compute per-profile compliance:
    fraction of time steps in each scenario where load >= bid
    """
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
    # Compute per-profile compliance
    also_comp = per_profile_compliance(x_also, out_sample)
    cvar_comp = per_profile_compliance(x_cvar, out_sample)

    # CDFs
    x1, y1 = empirical_cdf(also_comp)
    x2, y2 = empirical_cdf(cvar_comp)

    

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(x1, y1, label=f"ALSO-X ")
    plt.plot(x2, y2, label=f"CVaR ")

    # P90 horizontal line
    plt.axvline(0.9, linestyle="--", color="black", label="P90 threshold")


    plt.xlabel("Per-profile compliance")
    plt.ylabel("Empirical CDF")
    plt.title("Compliance Distribution (CDF of Per-Profile Compliance)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("Results","Step 2 Plots", "compliance_cdf.png"))

def plot_p90_sensitivity_panel(epsilons, bids,
                              sf_in, sf_out,
                              sf95_out,
                              violation_rate_in, violation_rate_out):

    

    save_dir = os.path.join("Results", "Step 2 Plots")
    os.makedirs(save_dir, exist_ok=True)

    reliability = (1 - epsilons) * 100

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    # --- 1. Average shortfall (IN vs OUT) ---
    axes[0, 0].plot(reliability, sf_in, marker='o', label="In-sample")
    axes[0, 0].plot(reliability, sf_out, marker='s', linestyle='--', label="Out-of-sample")
    axes[0, 0].set_title("Average Shortfall")
    axes[0, 0].set_ylabel("kW")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # --- 2. Violation rate (IN vs OUT) ---
    axes[0, 1].plot(reliability, violation_rate_in, marker='o', label="In-sample")
    axes[0, 1].plot(reliability, violation_rate_out, marker='s', linestyle='--', label="Out-of-sample")
    axes[0, 1].set_title("Violation Rate")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # --- 3. Optimal bid ---
    axes[1, 0].plot(reliability, bids, marker='o')
    axes[1, 0].set_title("Optimal Bid")
    axes[1, 0].set_xlabel("Reliability Level (%)")
    axes[1, 0].set_ylabel("kW")
    axes[1, 0].grid(True)

    # --- 4. 95th percentile shortfall (OUT only) ---
    axes[1, 1].plot(reliability, sf95_out, marker='o')
    axes[1, 1].set_title("95th Percentile Shortfall (Out-of-sample)")
    axes[1, 1].set_xlabel("Reliability Level (%)")
    axes[1, 1].set_ylabel("kW")
    axes[1, 1].grid(True)

    # Invert x-axis (optional but recommended)
    for ax in axes.flat:
        ax.invert_xaxis()

    fig.suptitle("Impact of Reliability Requirement on ALSO-X (In vs Out-of-Sample)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(save_dir, "p90_sensitivity_panel.png"), dpi=300)
    plt.close()

if __name__ == "__main__":

    data = np.load(os.path.join("Results", "Step 2","bids_results.npz"))
    
    x_also = data["x_also"]
    x_cvar = data["x_cvar"]
    out_sample = data["out_sample"]
    

    import pandas as pd

    table = pd.read_csv(os.path.join("Results", "Step 2", "p90_analysis_table.csv"))

    plot_p90_sensitivity_panel(
    table["epsilon"].values,
    table["bid"].values,
    table["sf_in"].values,
    table["sf_out"].values,
    table["sf95_out"].values,
    table["violation_rate_in (%)"].values,
    table["violation_rate_out (%)"].values
)