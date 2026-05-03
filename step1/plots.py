"""
step1/plots.py
Reusable plotting functions for Step 1 tasks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text


def plot_profit_histogram(scenario_profit, prob, title=None, save_path=None, color="#FF6B35"):
    """
    Histogram of per-scenario profits with a vertical line at the expected profit.

    Parameters
    ----------
    scenario_profit : dict {omega: float}
    prob            : Series of scenario probabilities (indexed by scenario ID)
    title           : str – plot title
    save_path       : Path-like – file to save the figure to
    color           : str – bar colour
    """
    profits = list(scenario_profit.values())
    scenarios = list(scenario_profit.keys())
    expected = sum(prob[omega] * scenario_profit[omega] for omega in scenarios)

    plt.figure(figsize=(10, 5))
    plt.hist(profits, bins=50, color=color, edgecolor='white')
    plt.axvline(
        x=expected, color='red', linestyle='--', linewidth=2,
        label=f' Total expected profit: €{expected:,.0f}'
    )
    plt.legend()
    plt.xlabel("Profit (€)")
    plt.ylabel("Number of scenarios")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_Expected_DA_And_Balancing_Values(lambda_Balancing, lambda_One_Price_DA, hours, save_path, color="#fa9537"):
    """
    Plot of expected day-ahead value, expected balancing value and the difference for each hour to determine the decision of how much to offer in the day-ahead market.
    """
    hours = list(hours)

    plt.figure(figsize=(10, 5))
    plt.plot(hours, [lambda_One_Price_DA[h] for h in hours], marker='o', color='steelblue', label='Expected One-Price DA Value')
    plt.plot(hours, [lambda_Balancing[h] for h in hours], marker='s', color='darkgreen', label='Expected One-Price Balancing Value')
    plt.xlabel('Hour of Day')
    plt.ylabel('Value (€/MWh)')
    plt.title('Expected Day-Ahead price vs expected balancing price')
    plt.xticks(hours, [h + 1 for h in hours])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_Mean_Wind_And_DA_Offers(wind_mw, p_DA_one, p_DA_two, hours, save_path=None, color="#fa9537"):
    """
    Plot of mean wind generation and optimal day-ahead offers (one-price vs two-price).

    Parameters
    ----------
    wind_mw   : DataFrame of wind power realisations (scenarios × hours)
    p_DA_one  : dict {h: float} - one-price DA offers
    p_DA_two  : dict {h: float} - two-price DA offers
    hours     : list of hour IDs
    save_path : Path-like – file to save the figure to
    color     : str – line colour for wind generation
    """
    hours      = list(hours)
    mean_wind  = wind_mw[hours].mean()
    one_offers = [p_DA_one[h] for h in hours]
    two_offers = [p_DA_two[h] for h in hours]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Power (MW)')
    ax1.plot(hours, mean_wind,  marker='o', color=color,       label='Mean Wind Generation (MW)')
    ax1.plot(hours, one_offers, marker='s', color='steelblue', label='One-Price DA Offer (MW)')
    ax1.plot(hours, two_offers, marker='^', color='darkgreen', label='Two-Price DA Offer (MW)')
    ax1.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, borderaxespad=0)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([h + 1 for h in hours])

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_hourly_offers(p_DA_values, hours, save_path, title, color="#2196F3"):
    """
    Bar plot of optimal day-ahead offers for each hour.
    """
    hours = list(hours)
    offers = [p_DA_values[h] for h in hours]

    plt.figure(figsize=(10, 5))
    plt.bar(hours, offers, color=color, edgecolor='black', linewidth=0.5)
    plt.xlabel('Hour of Day')
    plt.ylabel('Offer (MW)')
    plt.title('Optimal day-ahead offers')
    plt.title(title)
    plt.xticks(hours, [h + 1 for h in hours])
    plt.ylim(0, 600)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_cvar_frontier(frontier_df, title, save_path):
    """
    Scatter/line plot of expected profit vs CVaR for a beta sweep.

    Parameters
    ----------
    frontier_df : DataFrame with columns ['beta', 'expected_profit', 'cvar']
    title       : str – plot title
    save_path   : Path-like – file to save the figure to
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frontier_df["cvar"], frontier_df["expected_profit"],
             marker="o", color="#1f77b4")
    for _, row in frontier_df.iterrows():
        plt.annotate(
            f"beta={row['beta']:.2f}",
            (row["cvar"], row["expected_profit"]),
            textcoords="offset points", xytext=(5, 5)
        )
    plt.xlabel("CVaR (alpha = 0.90)")
    plt.ylabel("Expected Profit (€)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_cvar_frontier_With_Both_Models(frontier_single_price_df, frontier_two_price_df, title=None, save_path=None):
    """
    Scatter/line plot of expected profit vs CVaR for a beta sweep.

    Parameters
    ----------
    frontier_single_price_df : DataFrame with columns ['beta', 'expected_profit', 'cvar']
    frontier_two_price_df    : DataFrame with columns ['beta', 'expected_profit', 'cvar']
    title                    : str – plot title
    save_path                : Path-like – file to save the figure to
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frontier_single_price_df["cvar"], frontier_single_price_df["expected_profit"],
             marker="o", color="#1f77b4", label="One-Price Model")
    plt.plot(frontier_two_price_df["cvar"], frontier_two_price_df["expected_profit"],
             marker="s", color="#ff7f0e", label="Two-Price Model")

    # annotate first and last point of the one-price line
    first_one = frontier_single_price_df.iloc[0]
    last_one  = frontier_single_price_df.iloc[-1]
    plt.annotate(
        f"$\\beta=0$",
        (first_one["cvar"], first_one["expected_profit"]),
        textcoords="offset points", xytext=(5, 5),
        fontsize=9
    )
    plt.annotate(
        f"$\\beta=1$",
        (last_one["cvar"], last_one["expected_profit"]),
        textcoords="offset points", xytext=(5, 5),
        fontsize=9
    )

    # annotate all points of the two-price line
    offset = 25
    for _, row in frontier_two_price_df.iterrows():
        plt.annotate(
            f"$\\beta={row['beta']:.2f}$",
            (row["cvar"], row["expected_profit"]),
            textcoords="offset points", xytext=(5, offset),
            fontsize=9
        )
        offset -= 5

    plt.xlabel("CVaR (€), $\\alpha = 0.90$")
    plt.ylabel("Expected Profit (€)")
    plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, borderaxespad=0)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_crossvalidation(results_df, save_path_scatter=None, save_path_bar=None):
    """
    Two plots for k-fold cross-validation results:
      1. Scatter: in-sample vs out-of-sample profit per fold
      2. Bar chart: in-sample vs out-of-sample profit per fold

    Parameters
    ----------
    results_df        : DataFrame with columns
                        ['fold', 'insample_profit', 'outsample_profit']
    save_path_scatter : Path-like
    save_path_bar     : Path-like
    """
    n_folds = len(results_df)

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(results_df['insample_profit'], results_df['outsample_profit'],
               color='steelblue', s=100, zorder=5, label='Fold')

    min_val = min(results_df['insample_profit'].min(), results_df['outsample_profit'].min())
    max_val = max(results_df['insample_profit'].max(), results_df['outsample_profit'].max())
    margin  = (max_val - min_val) * 0.5
    ax.plot(
        [min_val - margin * 0.7, max_val + margin * 0.7],
        [min_val - margin * 0.7, max_val + margin * 0.7],
        'r--', linewidth=1.5, label='Perfect generalization'
    )

    texts = []
    for _, row in results_df.iterrows():
        texts.append(ax.annotate(
            f"Fold {int(row['fold'] + 1)}",
            (row['insample_profit'], row['outsample_profit']),
            fontsize=12
        ))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_xlabel("In-sample profit (€)")
    ax.set_ylabel("Out-of-sample profit (€)")
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, borderaxespad=0)
    plt.tight_layout()
    if save_path_scatter is not None:
        plt.savefig(save_path_scatter, dpi=150)
    plt.show()

    # --- Bar plot ---
    x     = np.arange(n_folds)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, results_df['insample_profit'],  width,
           label='In-sample',     color='steelblue')
    ax.bar(x + width / 2, results_df['outsample_profit'], width,
           label='Out-of-sample', color='#FF6B35')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Profit (€)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i + 1}' for i in range(n_folds)])
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, borderaxespad=0)
    plt.tight_layout()
    if save_path_bar is not None:
        plt.savefig(save_path_bar, dpi=150)
    plt.show()

def plot_hourly_offers_and_prices(p_DA_values, lambda_Balancing, lambda_DA, hours, save_path, title=None, 
                                   color_offer="#FF6B35", color_DA="steelblue", color_balancing="#4CAF50"):
    """
    Combined plot: bar chart of optimal day-ahead offers (left axis) and
    expected DA vs balancing prices (right axis) for each hour.
    """
    hours = list(hours)
    offers = [p_DA_values[h] for h in hours]
    da_prices = [lambda_DA[h] for h in hours]
    bal_prices = [lambda_Balancing[h] for h in hours]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # left axis: hourly offers
    ax1.bar(hours, offers, color=color_offer, edgecolor='black', linewidth=0.5, alpha=0.7, label='Day-Ahead Offer (MW)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Offer (MW)', color=color_offer)
    ax1.tick_params(axis='y', labelcolor=color_offer)
    ax1.set_ylim(0, 600)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([h + 1 for h in hours])

    # right axis: expected prices
    ax2 = ax1.twinx()
    ax2.plot(hours, da_prices, marker='o', color=color_DA, label='Expected DA Price (€/MWh)')
    ax2.plot(hours, bal_prices, marker='s', color=color_balancing, label='Expected Balancing Price (€/MWh)')
    ax2.set_ylabel('Price (€/MWh)')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.legend(lines1 + lines2, labels1 + labels2,
           bbox_to_anchor=(0.5, 1.02), loc='lower center',
           ncol=3, borderaxespad=0, fontsize=9)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_profit_histogram_comparison(scenario_profit_one, scenario_profit_two, prob,
                                      title=None, save_path=None,
                                      color_one="#FF6B35", color_two="#4CAF50"):
    """
    Overlapping histogram of per-scenario profits for one-price and two-price schemes.

    Parameters
    ----------
    scenario_profit_one : dict {omega: float} - one-price profits
    scenario_profit_two : dict {omega: float} - two-price profits
    prob                : Series of scenario probabilities
    title               : str – plot title
    save_path           : Path-like – file to save the figure to
    color_one           : str – bar colour for one-price
    color_two           : str – bar colour for two-price
    """
    scenarios    = list(scenario_profit_one.keys())
    profits_one  = list(scenario_profit_one.values())
    profits_two  = list(scenario_profit_two.values())
    expected_one = sum(prob[omega] * scenario_profit_one[omega] for omega in scenarios)
    expected_two = sum(prob[omega] * scenario_profit_two[omega] for omega in scenarios)

    # common bin edges
    all_profits  = profits_one + profits_two
    _, bin_edges = np.histogram(all_profits, bins=50)

    plt.figure(figsize=(10, 5))
    plt.hist(profits_one, bins=bin_edges, color=color_one, edgecolor='white',
             alpha=0.6, label='One-Price')
    plt.hist(profits_two, bins=bin_edges, color=color_two, edgecolor='white',
             alpha=0.6, label='Two-Price')
    plt.axvline(x=expected_one, color=color_one, linestyle='--', linewidth=2,
                label=f'E[profit] one-price: €{expected_one:,.0f}')
    plt.axvline(x=expected_two, color=color_two, linestyle='--', linewidth=2,
                label=f'E[profit] two-price: €{expected_two:,.0f}')
    plt.xlabel('Profit (€)')
    plt.ylabel('Number of scenarios')
    plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, borderaxespad=0)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_offers_and_prices_two(wind_mw, p_DA_one, p_DA_two, 
                                expected_DA, expected_B_up, expected_B_down,
                                hours, save_path=None,
                                color_wind="#fa9537", color_one="steelblue", 
                                color_two="darkgreen"):
    """
    Two-subplot figure:
    - Top: mean wind generation + one-price and two-price DA offers (MW)
    - Bottom: expected DA, balancing up and balancing down prices (€/MWh)

    Parameters
    ----------
    wind_mw        : DataFrame of wind power realisations (scenarios × hours)
    p_DA_one       : dict {h: float} - one-price DA offers
    p_DA_two       : dict {h: float} - two-price DA offers
    expected_DA    : dict {h: float} - expected DA price
    expected_B_up  : dict {h: float} - expected balancing up price
    expected_B_down: dict {h: float} - expected balancing down price
    hours          : list of hour IDs
    save_path      : Path-like – file to save the figure to
    """
    hours      = list(hours)
    mean_wind  = wind_mw[hours].mean()
    one_offers = [p_DA_one[h] for h in hours]
    two_offers = [p_DA_two[h] for h in hours]
    da_prices  = [expected_DA[h]     for h in hours]
    b_up       = [expected_B_up[h]   for h in hours]
    b_down     = [expected_B_down[h] for h in hours]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # top subplot: offers
    ax1.plot(hours, mean_wind,  marker='o', color=color_wind, label='Mean Wind Generation (MW)')
    ax1.plot(hours, one_offers, marker='s', color=color_one,  label='One-Price DA Offer (MW)')
    ax1.plot(hours, two_offers, marker='^', color=color_two,  label='Two-Price DA Offer (MW)')
    ax1.set_ylabel('Power (MW)')
    ax1.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, borderaxespad=0)

    # bottom subplot: prices
    ax2.plot(hours, da_prices, marker='o', color=color_one,  label='Expected DA Price (€/MWh)')
    ax2.plot(hours, b_up,      marker='s', color=color_two,  label='Expected Balancing Up Price (€/MWh)')
    ax2.plot(hours, b_down,    marker='^', color='firebrick', label='Expected Balancing Down Price (€/MWh)')
    ax2.set_ylabel('Price (€/MWh)')
    ax2.set_xlabel('Hour of Day')
    ax2.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, borderaxespad=0)
    ax2.set_xticks(hours)
    ax2.set_xticklabels([h + 1 for h in hours])

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show() 

def plot_profit_boxplot(frontier, color, title, save_path, alpha=0.9):
    betas = [row["beta"] for row in frontier]
    data  = [list(row["scenario_profit"].values()) for row in frontier]

    fig, ax = plt.subplots(figsize=(9, 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, color=color),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # VaR line only — no E[Π] line
    vars_ = [
        float(np.quantile(list(row["scenario_profit"].values()), 1 - alpha))
        for row in frontier
    ]
    ax.plot(
        range(1, len(betas) + 1), vars_,
        marker="s", markersize=4, linewidth=1.2,
        linestyle="--", color="gray", label=rf"VaR ($\alpha={alpha}$)"
    )

    ax.set_xticks(range(1, len(betas) + 1))
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Pi_\omega$ [€]")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}") 

def plot_profit_boxplot_comparison(frontier_one, frontier_two, save_path, alpha=0.9):
    betas = [row["beta"] for row in frontier_one]
    data_one = [list(row["scenario_profit"].values()) for row in frontier_one]
    data_two = [list(row["scenario_profit"].values()) for row in frontier_two]

    n      = len(betas)
    x      = np.arange(n)
    width  = 0.35
    offset = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))

    def draw_boxes(data, positions, color, label):
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=2, alpha=0.3, color=color),
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        # proxy artist for legend
        ax.plot([], [], color=color, linewidth=6, alpha=0.4, label=label)

    draw_boxes(data_one, x - offset, "#fa9537", "one-price")
    draw_boxes(data_two, x + offset, "#3fe60c", "two-price")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Pi_\omega$ [€]")
    #ax.set_title("Profit distribution vs. risk aversion — one-price vs. two-price")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}") 

def plot_hourly_offers(frontier_two, wind_mw, si, scenarios, save_path):
    hours         = list(frontier_two[0]["p_DA_values"].keys())
    hours_display = [h + 1 for h in hours]
    ref_wind      = [wind_mw[h].mean() for h in hours]

    max_offer = max(max(row["p_DA_values"][h] for h in hours) for row in frontier_two)
    colors    = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#a50026"]

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax2 = ax1.twinx()

    for row, c in zip(frontier_two, colors):
        offers = [row["p_DA_values"][h] for h in hours]
        ax1.plot(hours_display, offers, marker="o", markersize=3, linewidth=1.4,
                 color=c, label=rf"$\beta={row['beta']}$")

    ax1.plot(hours_display, ref_wind, linestyle="--", linewidth=1.2,
             color="black", label=r"$\mathbb{E}[p^{\mathrm{real}}_{t,\omega}]$")

    ax1.set_xlabel("hour")
    ax1.set_ylabel(r"$p_t^{\mathrm{DA}}$ [MW]")
    ax1.set_xticks(hours_display)
    ax1.set_ylim(0, max_offer * 1.15)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel(r"$P(\mathrm{SI}_{t,\omega})$")
    ax2.spines[["top", "left"]].set_visible(False)

    ax1.legend(frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")