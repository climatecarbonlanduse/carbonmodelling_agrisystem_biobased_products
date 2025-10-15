
"""
Final net CO₂e grid with two filled fields (storage and emissions) and net line.
Reads carbon_storage_data.pkl and emissions_data.pkl; writes final_net_impact_data.pkl.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def final_stacked_net_all():
    # ---------------------------- File paths ----------------------------
    carbon_pkl = Path("carbon_storage_data.pkl")
    emissions_pkl = Path("emissions_data.pkl")
    out_png = Path("enter path here")  # output figure
    out_final_pkl = Path("final_net_impact_data.pkl")
    out_excel = Path("enter path here")  # net_carbon_post_3x6_save.xlsx
    out_pickle_flat = Path("enter path here")  # net_carbon_post_3x6_save.pkl

    # -------------------------- Load precomputed -------------------------
    with carbon_pkl.open("rb") as f:
        carbon_data = pickle.load(f)
    with emissions_pkl.open("rb") as f:
        emissions_data = pickle.load(f)

    scenarios = ["Scenario1", "Scenario2", "Scenario3"]
    products_all = ["isolation", "cellulose", "biochar", "methane", "longplastic", "shortplastic"]

    # Plot order (isolation excluded)
    plot_products = ["shortplastic", "longplastic", "methane", "cellulose", "biochar"]

    product_names = {
        "isolation":    "Isolation",
        "cellulose":    "Cellulose Products",
        "biochar":      "Biochar",
        "methane":      "Biomethane",
        "longplastic":  "Bioplastics",
        "shortplastic": "Bioplastics",
    }
    custom_row_labels = {
        "isolation":    "Insulation\nton CO₂eq/ha",
        "shortplastic": "Short-lived\nBioplastics\nton CO₂eq/ha",
        "longplastic":  "Long-lived\nBioplastics\nton CO₂eq/ha",
        "methane":      "Biomethane\nton CO₂eq/ha",
        "cellulose":    "Cellulose\nProducts\nton CO₂eq/ha",
        "biochar":      "Biochar\nton CO₂eq/ha",
    }

    # ---------------------------- Figure set-up --------------------------
    fig, axes = plt.subplots(len(plot_products), len(scenarios),
                             figsize=(10, 8), sharex=True, sharey=False)
    if len(plot_products) == 1:
        axes = axes.reshape(1, -1)

    color_net = "darkorange"
    color_storage = "darkslategrey"
    color_emissions = "seagreen"
    alpha_fill = 0.7

    # ---------------------------- Main loop ------------------------------
    for i, product in enumerate(plot_products):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            try:
                years = carbon_data[product][scenario]["Year"]
                years_array = np.asarray(years, dtype=int)

                storage_vals = np.asarray(carbon_data[product][scenario]["carbon_stored_total"], dtype=float)
                emissions_vals = np.asarray(emissions_data[product][scenario]["net"], dtype=float)
                net_impact = storage_vals + emissions_vals

                # Fill: storage
                ax.fill_between(years_array, 0, storage_vals,
                                alpha=alpha_fill, facecolor=color_storage, edgecolor="none",
                                zorder=1, label="C in additional SOC and products")
                # Fill: emissions (stacked on storage)
                ax.fill_between(years_array, storage_vals, net_impact,
                                alpha=alpha_fill, facecolor=color_emissions, edgecolor="none",
                                zorder=1, label="Cumulative net GHG emissions")

                # Net line
                ax.plot(years_array, net_impact, color=color_net, linewidth=2.5, linestyle="-", zorder=3)

                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=0)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.tick_params(axis="both", labelsize=10)
                ax.set_ylim(-210, 20)

                if i == 0:
                    ax.set_title(scenario, fontsize=13, fontweight="bold")
                if j == 0:
                    ax.text(-0.7, 0.5, custom_row_labels[product],
                            fontsize=13, fontweight="bold", linespacing=1.2,
                            va="center", ha="center", transform=ax.transAxes)
                if j == len(scenarios) - 1:
                    grp = "FP" if product == "cellulose" else "PY" if product == "biochar" else "AD"
                    ax.text(1.15, 0.5, grp, fontsize=14, fontweight="bold",
                            transform=ax.transAxes, ha="left", va="center")
            except KeyError:
                continue

    for ax in axes[-1]:
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.2, top=0.93)

    # Legend (two fields + net line)
    legend_handles = [
        Line2D([0], [0], color=color_net, linewidth=2.5, linestyle="-", label="Net"),
        Patch(facecolor=color_emissions, alpha=alpha_fill, label="Cumulative Net GHG Emissions"),
        Patch(facecolor=color_storage, alpha=alpha_fill, label="C in additional SOC and products"),
    ]
    leg = fig.legend(handles=legend_handles, loc="upper center", fontsize=12, ncol=3,
                     bbox_to_anchor=(0.5, 1.04), frameon=True)
    leg.get_frame().set_edgecolor("lightgray")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor("white")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------- Build & save net_impact_data ---------------------
    net_impact_data = {}
    for product in products_all:
        net_impact_data[product] = {}
        for scenario in scenarios:
            try:
                years_array = np.asarray(carbon_data[product][scenario]["Year"], dtype=int)
                storage_vals = np.asarray(carbon_data[product][scenario]["carbon_stored_total"], dtype=float)
                emissions_vals = np.asarray(emissions_data[product][scenario]["net"], dtype=float)
                net_vals = storage_vals + emissions_vals
                net_impact_data[product][scenario] = {
                    "Year": years_array.tolist(),
                    "Net_CO2e": net_vals.tolist(),
                }
            except KeyError:
                continue

    with out_final_pkl.open("wb") as f:
        pickle.dump(net_impact_data, f)

    # Optional flat export
    rows = []
    for prod, scen_dict in net_impact_data.items():
        for scen, dat in scen_dict.items():
            for y, v in zip(dat["Year"], dat["Net_CO2e"]):
                rows.append({"Product": prod, "Scenario": scen, "Year": y, "Net_CO2e_ton_ha": v})

    df_net = pd.DataFrame(rows)
    if out_excel.name != "enter path here":
        df_net.to_excel(out_excel, index=False)
    if out_pickle_flat.name != "enter path here":
        with out_pickle_flat.open("wb") as f:
            pickle.dump(df_net, f)

if __name__ == "__main__":
    final_stacked_net_all()
