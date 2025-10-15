
"""
Emissions and avoided emissions for biobased product scenarios.
Generates cumulative emissions plots by product and scenario.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pickle

# ------------------------------------------------------------------
# 1) Load avoided-emissions data (long-lived plastics)
# ------------------------------------------------------------------
release_path = Path("enter path here")  # Excel with cumulative release data
df_release = pd.read_excel(release_path)

# Columns with cumulative release data (t)
rel_cols = [
    "Cum Release 100y (t)",
    "Cum Release 30y (t)",
    "Cum Release Double Ley (t)"
]
for c in rel_cols:
    df_release[c] = (
        df_release[c].astype(str)
                     .str.replace(",", ".")
                     .astype(float)
    )

# Convert cumulative tonnes → yearly increments (kg y⁻¹, negative = avoided)
def cum_to_yearly_kg_neg(series_tonnes):
    yearly_tonnes = series_tonnes.diff().fillna(series_tonnes)
    return (-(yearly_tonnes * 1000)).tolist()

release_yearly_kg_neg = {
    "Scenario1": cum_to_yearly_kg_neg(df_release["Cum Release 100y (t)"]),
    "Scenario2": cum_to_yearly_kg_neg(df_release["Cum Release 30y (t)"]),
    "Scenario3": cum_to_yearly_kg_neg(df_release["Cum Release Double Ley (t)"])
}

# ------------------------------------------------------------------
# 2) Define products and placeholder parameters
# ------------------------------------------------------------------
years = list(range(2024, 2124))  # 100-year span

# Each field should be filled by the script operator
product_inputs = {
    "shortplastic": {
        "diesel_transport": "enter value here",
        "electricity_GB": "enter value here",
        "electricity_AD": "enter value here",
        "energy_production": "enter value here",
        "el_EF_factor": "enter value here",
        "avoided_emissions": "enter value here",
        "avoided_emissions_nitrogen_effect": "enter value here",
        "improvement": "enter value here",
        "nitrogen_improvement": "enter value here"
    },
    "longplastic": {
        "diesel_transport": "enter value here",
        "electricity_GB": "enter value here",
        "electricity_AD": "enter value here",
        "energy_production": "enter value here",
        "el_EF_factor": "enter value here",
        "avoided_emissions": "enter value here",
        "avoided_emissions_nitrogen_effect": "enter value here",
        "improvement": "enter value here",
        "nitrogen_improvement": "enter value here"
    },
    "methane": {
        "diesel_transport": "enter value here",
        "electricity_GB": "enter value here",
        "electricity_AD": "enter value here",
        "energy_production": "enter value here",
        "el_EF_factor": "enter value here",
        "avoided_emissions": "enter value here",
        "avoided_emissions_nitrogen_effect": "enter value here",
        "improvement": "enter value here",
        "nitrogen_improvement": "enter value here"
    },
    "cellulose": {
        "diesel_transport": "enter value here",
        "electricity_GB": "enter value here",
        "electricity_AD": "enter value here",
        "energy_production": "enter value here",
        "el_EF_factor": "enter value here",
        "avoided_emissions": "enter value here",
        "avoided_emissions_nitrogen_effect": "enter value here",
        "improvement": "enter value here",
        "nitrogen_improvement": "enter value here"
    },
    "biochar": {
        "diesel_transport": "enter value here",
        "electricity_GB": "enter value here",
        "electricity_AD": "enter value here",
        "energy_production": "enter value here",
        "el_EF_factor": "enter value here",
        "avoided_emissions": "enter value here",
        "avoided_emissions_nitrogen_effect": "enter value here",
        "improvement": "enter value here",
        "nitrogen_improvement": "enter value here"
    }
}

# ------------------------------------------------------------------
# 3) Scenario behavior
# ------------------------------------------------------------------
def apply_scenario_behavior(base_list, yrs, scenario):
    out = []
    for i, yr in enumerate(yrs):
        if scenario == "Scenario1":
            out.append(base_list[i] * (2 / 6) if yr <= yrs[0] + 29 else 0)
        elif scenario == "Scenario2":
            out.append(base_list[i] * (2 / 6))
        elif scenario == "Scenario3":
            if yr <= yrs[0] + 29:
                out.append(base_list[i] * (2 / 6))
            else:
                out.append(base_list[i] * (4 / 6))
    return out

# ------------------------------------------------------------------
# 4) Build emissions data
# ------------------------------------------------------------------
product_emissions = {}
for product, inputs in product_inputs.items():
    diesel_list, el1_list, ad_list, prod_list = [], [], [], []
    avoided_list, avoided_nitro_list = [], []

    for i in range(len(years)):
        imp = (1 - float(inputs["improvement"])) ** i
        imp_n = (1 - float(inputs["nitrogen_improvement"])) ** i

        diesel_list.append(float(inputs["diesel_transport"]) * imp)
        el1_list.append(float(inputs["electricity_GB"]) * imp)
        ad_list.append(float(inputs["electricity_AD"]) * imp)
        prod_list.append(float(inputs["energy_production"]) * imp)
        avoided_list.append(float(inputs["avoided_emissions"]) * imp)
        avoided_nitro_list.append(float(inputs["avoided_emissions_nitrogen_effect"]) * imp_n)

    base_emis = [a + b + c + d for a, b, c, d in zip(diesel_list, el1_list, ad_list, prod_list)]
    base_avo = [a - b for a, b in zip(avoided_list, avoided_nitro_list)]

    product_emissions[product] = {}
    for scen in ["Scenario1", "Scenario2", "Scenario3"]:
        if product == "longplastic":
            base_avo_adj = [ba + ea for ba, ea in zip(base_avo, release_yearly_kg_neg[scen])]
        else:
            base_avo_adj = base_avo

        emis_series = (np.cumsum(apply_scenario_behavior(base_emis, years, scen)) / 1000)
        avoid_series = (np.cumsum(apply_scenario_behavior(base_avo_adj, years, scen)) / 1000)

        product_emissions[product][scen] = {
            "Year": years,
            "emissions": emis_series,
            "avoided": avoid_series,
            "net": emis_series + avoid_series
        }

# ------------------------------------------------------------------
# 5) Plotting
# ------------------------------------------------------------------
def emissions_avoided_plot(prod_emis, row_labels):
    scenarios = ["Scenario1", "Scenario2", "Scenario3"]
    plot_products = ["shortplastic", "longplastic", "methane", "cellulose", "biochar"]

    fig, axes = plt.subplots(len(plot_products), len(scenarios),
                             figsize=(10, 8), sharex=True, sharey=False)

    for i, product in enumerate(plot_products):
        for j, scen in enumerate(scenarios):
            data = prod_emis[product][scen]
            yrs = data["Year"]
            emis = data["emissions"]
            avoid = data["avoided"]
            net = data["net"]

            ax = axes[i, j]
            ax.fill_between(yrs, 0, emis, color="firebrick", alpha=0.7)
            ax.fill_between(yrs, 0, avoid, color="seagreen", alpha=0.7)
            ax.plot(yrs, net, color="black", linestyle="--", linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", labelsize=10)

            if product in ["cellulose", "biochar"]:
                ax.set_ylim(-70, 50)
            else:
                ax.set_ylim(-140, 50)

            if i == 0:
                ax.set_title(scen, fontsize=13, fontweight="bold")

            if j == 0:
                ax.text(-0.5, 0.5, row_labels[product],
                        fontsize=13, fontweight="bold",
                        va="center", ha="center", linespacing=1.2,
                        transform=ax.transAxes)

            if j == len(scenarios) - 1:
                grp = "FP" if product == "cellulose" else "PY" if product == "biochar" else "AD"
                ax.text(1.11, 0.5, grp,
                        fontsize=14, fontweight="bold",
                        ha="left", va="center", transform=ax.transAxes)

    # Divider above last row (biochar)
    bc_idx = plot_products.index("cellulose")
    for ax in axes[bc_idx]:
        ax.plot([-0.1, 1.1], [1.11, 1.11], transform=ax.transAxes,
                color="orange", alpha=0.7, linestyle="--", linewidth=4.0, clip_on=False)

    for ax in axes[-1]:
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.2, top=0.93)

    fig.legend(handles=[
        Patch(facecolor="firebrick", alpha=0.7, label="Emissions"),
        Patch(facecolor="seagreen", alpha=0.7, label="Avoided Emissions"),
        Line2D([0], [0], color="black", linestyle="--", label="Net GHG Emissions")
    ], loc="upper center", fontsize=12, ncol=3, bbox_to_anchor=(0.5, 1.04))

    out_path = Path("enter path here")  # output figure path
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------------
# 6) Custom row labels and run
# ------------------------------------------------------------------
custom_row_labels = {
    "shortplastic": "Short-lived\nBioplastics\nton CO₂eq/ha",
    "longplastic": "Long-lived\nBioplastics\nton CO₂eq/ha",
    "methane": "Biomethane\nton CO₂eq/ha",
    "cellulose": "Cellulose\nProducts\nton CO₂eq/ha",
    "biochar": "Biochar\nton CO₂eq/ha"
}

emissions_avoided_plot(product_emissions, custom_row_labels)

# ------------------------------------------------------------------
# 7) Save emissions data
# ------------------------------------------------------------------
with open("emissions_data.pkl", "wb") as f:
    pickle.dump(product_emissions, f)
print("✅ emissions_data.pkl written")
