import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == "__main__":

    data = np.load(os.path.join("Data", "bids_results.npz"))

    x_also = data["x_also"]
    x_cvar = data["x_cvar"]
    out_sample = data["out_sample"]

    plot_compliance_cdf(x_also, x_cvar, out_sample)