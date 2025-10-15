
"""
Stacked SOC + product carbon storage by product and scenario.

Rows:
  1) Short-lived bioplastics (AD)
  2) Long-lived bioplastics (AD)
  3) Biomethane (AD)
  4) Cellulose products (FP)
  5) Biochar (PY)

Layers (bottom → top):
  • SOC in Baseline
  • Additional SOC in Scenario
  • Biogenic C stored in products
"""

from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def stacked_product_soc_plot():
    # ----------------------------- File paths -----------------------------
    base_folder = Path("enter path here")                     # e.g., .../data
    products_folder = base_folder / "products datasheets"     # folder of *.pkl
    agricultural_file = Path("enter path here")               # e.g., agrifloor.xlsx
    output_path = Path("enter path here")                     # figure PNG

    # ------------------------------ Load data -----------------------------
    pickle_files = [p for p in products_folder.glob("*.pkl")]
    product_dfs = {p.stem: pd.read_pickle(p) for p in pickle_files}
    df_agri = pd.read_excel(agricultural_file)

    # --------------------------- Column mappings --------------------------
    scenario_mapping = {
        "Scenario1": "Net Carbon 30y (tons)",
        "Scenario2": "Net Carbon 100y (tons)",
        "Scenario3": "Net Carbon Double Ley (tons)",
    }
    agri_columns = {
        "Scenario1": "Scenario1_MA",
        "Scenario2": "Scenario2_MA",
        "Scenario3": "Scenario3_MA",
    }
    baseline_column = "Baseline_MA"

    # ----------------------- Products and labels --------------------------
    product_names = {
        "isolation":    "Insulation Materials",
        "shortplastic": "Short-life Bioplastics",
        "longplastic":  "Long-life Bioplastics",
        "methane":      "Biomethane",
        "cellulose":    "Cellulose Products",
        "biochar":      "Biochar",
    }
    plot_products = ["shortplastic", "longplastic", "methane", "cellulose", "biochar"]
    scenarios = list(scenario_mapping.keys())

    custom_row_labels = {
        "shortplastic": "Short-lived\nBioplastics\nton CO₂eq/ha",
        "longplastic":  "Long-lived\nBioplastics\nton CO₂eq/ha",
        "methane":      "Biomethane\nton CO₂eq/ha",
        "cellulose":    "Cellulose\nProducts\nton CO₂eq/ha",
        "biochar":      "Biochar\nton CO₂eq/ha",
        "isolation":    "Insulation\nton CO₂eq/ha",
    }

    # --------------------------- Figure & axes ----------------------------
    fig, axes = plt.subplots(len(plot_products), len(scenarios),
                             figsize=(10, 9), sharex=True, sharey=False)
    if len(plot_products) == 1:
        axes = axes.reshape(1, -1)

    # ----------------------------- Plot loop ------------------------------
    for i, product in enumerate(plot_products):
        match = [k for k in product_dfs if product in k.lower()]
        if not match:
            continue
        df_prod = product_dfs[match[0]]

        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            product_col = scenario_mapping[scenario]
            soc_col = agri_columns[scenario]

            # Values (C → CO₂eq; minus = storage)
            years = df_prod["Year"].to_numpy()
            prod_vals = -1.0 * df_prod[product_col].to_numpy() * 3.67 * (2 / 6)
            baseline_vals = -1.0 * df_agri[baseline_column].iloc[:len(df_prod)].to_numpy() * 3.67 / 1000.0
            scenario_soc_vals = -1.0 * df_agri[soc_col].iloc[:len(df_prod)].to_numpy() * 3.67 / 1000.0
            additional_soc_vals = scenario_soc_vals - baseline_vals

            # Stacked areas
            ax.fill_between(years, 0, baseline_vals, color="#654321", alpha=0.9,
                            label="SOC in Baseline" if (i, j) == (0, 0) else None)
            ax.fill_between(years, baseline_vals, baseline_vals + additional_soc_vals,
                            color="saddlebrown", alpha=0.6,
                            label="Additional SOC in Scenario" if (i, j) == (0, 0) else None)
            if np.abs(prod_vals).sum() > 1e-6:
                ax.fill_between(years,
                                baseline_vals + additional_soc_vals,
                                baseline_vals + additional_soc_vals + prod_vals,
                                color="teal", alpha=0.8,
                                label="Biogenic C stored in products" if (i, j) == (0, 0) else None)

            # Row label
            if j == 0:
                ax.text(-0.5, 0.5, custom_row_labels[product],
                        fontsize=12, fontweight="bold", va="center", ha="center",
                        transform=ax.transAxes, linespacing=1)

            # Group tag
            if j == len(scenarios) - 1:
                group_label = "FP" if product == "cellulose" else "PY" if product == "biochar" else "AD"
                ax.text(1.11, 0.5, group_label, fontsize=14, fontweight="bold",
                        ha="left", va="center", transform=ax.transAxes)

            # Column title
            if i == 0:
                ax.set_title(scenario, fontsize=14, fontweight="bold")

            ax.tick_params(axis="both", labelsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Y-limits
            if product in ["shortplastic", "longplastic", "methane", "cellulose"]:
                ax.set_ylim(-75, 0)
            else:
                ax.set_ylim(-210, 0)

    # Common X label
    for ax in axes[-1]:
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")

    # Orange separator above PY row
    if "biochar" in plot_products:
        py_idx = plot_products.index("biochar")
        for ax in axes[py_idx]:
            ax.plot([-0.1, 1.1], [1.11, 1.11], transform=ax.transAxes,
                    color="orange", alpha=0.7, linestyle="--", linewidth=4.0,
                    clip_on=False)

    plt.subplots_adjust(wspace=0.25, hspace=0.25, top=0.92)

    # Legend (single, top-center)
    legend_elements = [
        Patch(facecolor="#654321", alpha=0.9, label="SOC in Baseline"),
        Patch(facecolor="saddlebrown", alpha=0.6, label="Additional SOC in Scenario"),
        Patch(facecolor="teal", alpha=0.8, label="Biogenic C stored in products"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, 1.04))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    stacked_product_soc_plot()
