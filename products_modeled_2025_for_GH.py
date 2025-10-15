

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
MODEL_YEARS_USE = 100 # set yourself  # extended horizon used by several scenarios

press_cake_biomass_input_base = 2000 # set by self 
press_cake_biomass_four_leys_base = 2 * press_cake_biomass_input_base


# -----------------------------------------------------------------------------
# Main container (defines scenario functions)
# -----------------------------------------------------------------------------
def all_parts_introduced_to_landscape_GB_AD_PY_and_process_carbon_flows():
    # -------------------------------------------------------------------------
    # Press cake → Insulation
    # -------------------------------------------------------------------------
    def press_cake_to_ISOLATION(biomass_utilization_factor=1.0, process_yield=1.0):
        MODEL_YEARS = 100
        production_stop_year = 30
        double_ley_start_year = 30
        time = np.arange(MODEL_YEARS)

        # Base parameters
        press_cake_biomass_input = press_cake_biomass_input_base
        press_cake_biomass_four_leys = press_cake_biomass_four_leys_base
        carbon_fraction_in_press_cake = 1.0
        carbon_in_biomass = 1.0
        grass_to_product_efficiency = 0.40  # placeholder efficiency

        # Biomass input by scenario
        annual_biomass_input_100y = np.full(MODEL_YEARS, press_cake_biomass_input)
        annual_biomass_input_30y = np.array([
            press_cake_biomass_input if yr < production_stop_year else 0 for yr in time
        ])
        annual_biomass_input_double_ley = np.array([
            press_cake_biomass_input if yr < double_ley_start_year else press_cake_biomass_four_leys
            for yr in time
        ])

        def calculate_carbon_dynamics(annual_biomass_input):
            effective_biomass_input = annual_biomass_input * biomass_utilization_factor * process_yield
            carbon_press_cake_per_year = (
                effective_biomass_input * carbon_in_biomass * carbon_fraction_in_press_cake * grass_to_product_efficiency
            )

            carbon_stored = np.zeros(MODEL_YEARS)
            carbon_released = np.zeros(MODEL_YEARS)

            # Storage accumulation
            for yr in range(MODEL_YEARS):
                carbon_stored[yr] = (carbon_stored[yr - 1] if yr else 0) + carbon_press_cake_per_year[yr]

            # Gaussian release
            mean_release_time = 52.5
            sd_release_time = 22.5
            for yr in range(MODEL_YEARS):
                for prev in range(yr + 1):
                    age = yr - prev
                    frac = (1 / (sd_release_time * np.sqrt(2 * np.pi))) * \
                           np.exp(-0.5 * ((age - mean_release_time) / sd_release_time) ** 2)
                    carbon_released[yr] += carbon_press_cake_per_year[prev] * frac

            net_seq = np.maximum(carbon_stored - np.cumsum(carbon_released), 0)
            return carbon_stored / 1000, carbon_released / 1000, net_seq / 1000  # kg → t

        # Run scenarios
        cS_100, cR_100, net_100 = calculate_carbon_dynamics(annual_biomass_input_100y)
        cS_30,  cR_30,  net_30  = calculate_carbon_dynamics(annual_biomass_input_30y)
        cS_DL,  cR_DL,  net_DL  = calculate_carbon_dynamics(annual_biomass_input_double_ley)

        # Export net carbon (optional)
        net_carbon_df = pd.DataFrame({
            "Year": time,
            "Net Carbon 100y (tons)": net_100,
            "Net Carbon 30y (tons)": net_30,
            "Net Carbon Double Ley (tons)": net_DL,
        })
        net_carbon_df.to_excel("enter path here", index=False)
        net_carbon_df.to_pickle("enter path here")

        # Plot
        colors = {
            "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759", "teal": "#76B7B2",
            "green": "#59A14F", "yellow": "#EDC948", "pink": "#FF9DA7",
            "brown": "#9C755F", "gray": "#BAB0AC"
        }

        plt.figure(figsize=(12, 6))

        # Panel 1: inputs, cumulative release, net
        plt.subplot(1, 2, 1)
        plt.plot(time, cS_30,  label="Scenario 1 C input", color=colors["gray"],  ls='-.', lw=3)
        plt.plot(time, cS_100, label="Scenario 2 C input", color=colors["brown"], ls='-.', lw=3)
        plt.plot(time, cS_DL,  label="Scenario 3 C input", color=colors["yellow"], ls='-.', lw=3)

        plt.plot(time, np.cumsum(cR_30),  label="Scenario 1 C Release", color=colors["teal"], ls='--', lw=3)
        plt.plot(time, np.cumsum(cR_100), label="Scenario 2 C Release", color=colors["pink"], ls='--', lw=3)
        plt.plot(time, np.cumsum(cR_DL),  label="Scenario 3 C Release", color=colors["red"],  ls='--', lw=3)

        plt.plot(time, net_30,  label="Scenario 1 Net C", color=colors["orange"], lw=3)
        plt.plot(time, net_100, label="Scenario 2 Net C", color=colors["green"],  lw=3)
        plt.plot(time, net_DL,  label="Scenario 3 Net C", color=colors["blue"],   lw=3)

        plt.xlabel("Year")
        plt.ylabel("Carbon Dynamics [t C ha⁻¹]")
        plt.legend(fontsize=11)
        plt.grid(True)

        # Panel 2: net only
        plt.subplot(1, 2, 2)
        plt.plot(time, net_30,  label="Scenario 1 Net C", color=colors["orange"], lw=4)
        plt.plot(time, net_100, label="Scenario 2 Net C", color=colors["green"],  lw=4)
        plt.plot(time, net_DL,  label="Scenario 3 Net C", color=colors["blue"],   lw=4)

        plt.xlabel("Year")
        plt.ylabel("Cumulative Net C Seq [t C ha⁻¹]")
        plt.legend(fontsize=11)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enter path here", dpi=300)
        plt.savefig("enter path here", dpi=600)
        # plt.show()

    # -------------------------------------------------------------------------
    # Press cake → Cellulose packaging (short-lived material)
    # -------------------------------------------------------------------------
    def press_cake_to_CELLULOSE_PACKAGING(biomass_utilization_factor=1.0, process_yield=1.0):
        SIMULATION_YEARS = MODEL_YEARS_USE
        production_stop_year = 30
        double_ley_start_year = 30
        time = np.arange(SIMULATION_YEARS)

        press_cake_biomass_input = press_cake_biomass_input_base
        press_cake_biomass_four_leys = press_cake_biomass_four_leys_base
        carbon_fraction_in_press_cake = 1.0
        carbon_in_biomass = 1.0
        grass_to_product_efficiency = 0.65

        annual_biomass_input_100y = np.full(SIMULATION_YEARS, press_cake_biomass_input)
        annual_biomass_input_30y = np.array([
            press_cake_biomass_input if yr < production_stop_year else 0 for yr in time
        ])
        annual_biomass_input_double_ley = np.array([
            press_cake_biomass_input if yr < double_ley_start_year else press_cake_biomass_four_leys
            for yr in time
        ])

        def calculate_carbon_dynamics(annual_biomass_input):
            effective_biomass_input = annual_biomass_input * biomass_utilization_factor * process_yield
            carbon_press_cake_per_year = (
                effective_biomass_input * carbon_in_biomass * carbon_fraction_in_press_cake * grass_to_product_efficiency
            )

            carbon_stored = np.zeros(SIMULATION_YEARS)
            carbon_released = np.zeros(SIMULATION_YEARS)

            for yr in range(SIMULATION_YEARS):
                carbon_stored[yr] = (carbon_stored[yr - 1] if yr else 0) + carbon_press_cake_per_year[yr]

            mean_release_time = 2
            sd_release_time = 1

            # Normalization per input year so release integrates to 1 over horizon
            normalization = np.zeros(SIMULATION_YEARS)
            for start in range(SIMULATION_YEARS):
                τ = np.arange(SIMULATION_YEARS - start)
                pdf = np.exp(-0.5 * ((τ - mean_release_time) / sd_release_time) ** 2) / (sd_release_time * np.sqrt(2 * np.pi))
                normalization[start] = pdf.sum()

            for yr in range(SIMULATION_YEARS):
                for prev in range(yr + 1):
                    age = yr - prev
                    frac = (np.exp(-0.5 * ((age - mean_release_time) / sd_release_time) ** 2) /
                            (sd_release_time * np.sqrt(2 * np.pi))) / normalization[prev]
                    carbon_released[yr] += carbon_press_cake_per_year[prev] * frac

            net_seq = np.maximum(carbon_stored - np.cumsum(carbon_released), 0)
            return carbon_stored / 1000, carbon_released / 1000, net_seq / 1000

        cS_100, cR_100, net_100 = calculate_carbon_dynamics(annual_biomass_input_100y)
        cS_30,  cR_30,  net_30  = calculate_carbon_dynamics(annual_biomass_input_30y)
        cS_DL,  cR_DL,  net_DL  = calculate_carbon_dynamics(annual_biomass_input_double_ley)

        plot_years = 100
        time_plot = np.arange(plot_years)

        net_carbon_df = pd.DataFrame({
            "Year": time_plot,
            "Net Carbon 100y (tons)": net_100[:plot_years],
            "Net Carbon 30y (tons)": net_30[:plot_years],
            "Net Carbon Double Ley (tons)": net_DL[:plot_years],
        })
        net_carbon_df.to_excel("enter path here", index=False)
        net_carbon_df.to_pickle("enter path here")

        colors = {
            "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759", "teal": "#76B7B2",
            "green": "#59A14F", "yellow": "#EDC948", "pink": "#FF9DA7",
            "brown": "#9C755F", "gray": "#BAB0AC"
        }

        plt.figure(figsize=(12, 6))

        # Panel 1
        plt.subplot(1, 2, 1)
        plt.plot(time_plot, cS_30[:plot_years],  label="Scenario 1 C input", color=colors["gray"],  ls='-.', lw=3)
        plt.plot(time_plot, cS_100[:plot_years], label="Scenario 2 C input", color=colors["brown"], ls='-.', lw=3)
        plt.plot(time_plot, cS_DL[:plot_years],  label="Scenario 3 C input", color=colors["yellow"], ls='-.', lw=3)

        plt.plot(time_plot, np.cumsum(cR_30)[:plot_years],  label="Scenario 1 C Release", color=colors["teal"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_100)[:plot_years], label="Scenario 2 C Release", color=colors["pink"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_DL)[:plot_years],  label="Scenario 3 C Release", color=colors["red"],  ls='--', lw=3)

        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=3)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=3)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=3)

        plt.xlabel("Year")
        plt.ylabel("Carbon Dynamics [t C ha⁻¹]")
        plt.title("Cellulose product – all scenarios (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        # Panel 2
        plt.subplot(1, 2, 2)
        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=5)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=5)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=5)

        plt.xlabel("Year")
        plt.ylabel("Cumulative Net C Seq [t C ha⁻¹]")
        plt.title("Net carbon sequestration – cellulose (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enter path here", dpi=300)
        plt.savefig("enter path here", dpi=600)
        # plt.show()

    # -------------------------------------------------------------------------
    # Press cake → Biochar (long-lived)
    # -------------------------------------------------------------------------
    def press_cake_to_BIOCHAR(biomass_utilization_factor=1.0, process_yield=1.0):
        SIMULATION_YEARS = MODEL_YEARS_USE
        production_stop_year = 30
        double_ley_start_year = 30
        time = np.arange(SIMULATION_YEARS)

        press_cake_biomass_input = press_cake_biomass_input_base
        press_cake_biomass_four_leys = press_cake_biomass_four_leys_base
        carbon_fraction_in_press_cake = 1.0
        carbon_in_biomass = 1.0

        # Biochar factors
        char_yield = 0.59
        carbon_content_bc = 0.64 * 0.90

        annual_biomass_input_100y = np.full(SIMULATION_YEARS, press_cake_biomass_input)
        annual_biomass_input_30y = np.array([
            press_cake_biomass_input if yr < production_stop_year else 0 for yr in time
        ])
        annual_biomass_input_double_ley = np.array([
            press_cake_biomass_input if yr < double_ley_start_year else press_cake_biomass_four_leys
            for yr in time
        ])

        def calculate_carbon_dynamics(annual_biomass_input):
            effective_in = annual_biomass_input * biomass_utilization_factor * process_yield
            carbon_press_cake_per_year = effective_in * carbon_in_biomass * carbon_fraction_in_press_cake * char_yield * carbon_content_bc

            carbon_stored = np.zeros(SIMULATION_YEARS)
            carbon_released = np.zeros(SIMULATION_YEARS)

            for yr in range(SIMULATION_YEARS):
                carbon_stored[yr] = (carbon_stored[yr - 1] if yr else 0) + carbon_press_cake_per_year[yr]

            mean_release_time = 100
            sd_release_time = 10

            normalization = np.zeros(SIMULATION_YEARS)
            for start in range(SIMULATION_YEARS):
                τ = np.arange(SIMULATION_YEARS - start)
                pdf = np.exp(-0.5 * ((τ - mean_release_time) / sd_release_time) ** 2) / (sd_release_time * np.sqrt(2 * np.pi))
                normalization[start] = pdf.sum()

            for yr in range(SIMULATION_YEARS):
                for prev in range(yr + 1):
                    age = yr - prev
                    frac = (np.exp(-0.5 * ((age - mean_release_time) / sd_release_time) ** 2) /
                            (sd_release_time * np.sqrt(2 * np.pi))) / normalization[prev]
                    carbon_released[yr] += carbon_press_cake_per_year[prev] * frac

            net_seq = np.maximum(carbon_stored - np.cumsum(carbon_released), 0)
            return carbon_stored / 1000, carbon_released / 1000, net_seq / 1000

        cS_100, cR_100, net_100 = calculate_carbon_dynamics(annual_biomass_input_100y)
        cS_30,  cR_30,  net_30  = calculate_carbon_dynamics(annual_biomass_input_30y)
        cS_DL,  cR_DL,  net_DL  = calculate_carbon_dynamics(annual_biomass_input_double_ley)

        plot_years = 100
        time_plot = np.arange(plot_years)

        net_carbon_df = pd.DataFrame({
            "Year": time_plot,
            "Net Carbon 100y (tons)": net_100[:plot_years],
            "Net Carbon 30y (tons)": net_30[:plot_years],
            "Net Carbon Double Ley (tons)": net_DL[:plot_years],
        })
        net_carbon_df.to_excel("enter path here", index=False)
        net_carbon_df.to_pickle("enter path here")

        colors = {
            "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759", "teal": "#76B7B2",
            "green": "#59A14F", "yellow": "#EDC948", "pink": "#FF9DA7",
            "brown": "#9C755F", "gray": "#BAB0AC"
        }

        plt.figure(figsize=(12, 6))

        # Panel 1
        plt.subplot(1, 2, 1)
        plt.plot(time_plot, cS_30[:plot_years],  label="Scenario 1 C input", color=colors["gray"],  ls='-.', lw=3)
        plt.plot(time_plot, cS_100[:plot_years], label="Scenario 2 C input", color=colors["brown"], ls='-.', lw=3)
        plt.plot(time_plot, cS_DL[:plot_years],  label="Scenario 3 C input", color=colors["yellow"], ls='-.', lw=3)

        plt.plot(time_plot, np.cumsum(cR_30)[:plot_years],  label="Scenario 1 C Release", color=colors["teal"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_100)[:plot_years], label="Scenario 2 C Release", color=colors["pink"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_DL)[:plot_years],  label="Scenario 3 C Release", color=colors["red"],  ls='--', lw=3)

        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=3)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=3)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=3)

        plt.xlabel("Year")
        plt.ylabel("Carbon Dynamics [t C ha⁻¹]")
        plt.title("Biochar – all scenarios (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        # Panel 2
        plt.subplot(1, 2, 2)
        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=5)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=5)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=5)

        plt.xlabel("Year")
        plt.ylabel("Cumulative Net C Seq [t C ha⁻¹]")
        plt.title("Net carbon sequestration – biochar (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enter path here", dpi=300)
        plt.savefig("enter path here", dpi=600)
        # plt.show()

    # -------------------------------------------------------------------------
    # CH4 → short-lived plastics
    # -------------------------------------------------------------------------
    def methane_to_plastic_SHORTTERM(biomass_utilization_factor=1.0, process_yield=1.0):
        SIMULATION_YEARS = MODEL_YEARS_USE
        production_stop_year = 30
        double_ley_start_year = 30
        time = np.arange(SIMULATION_YEARS)

        press_cake_biomass_input = press_cake_biomass_input_base
        press_cake_biomass_four_leys = press_cake_biomass_four_leys_base
        carbon_ADPC = 0.96
        biogas_yield = 0.36
        biomethane_to_C2_to_PHB = 0.50
        LOSS_FACTOR = 0.98  # (1 – 0.02)

        annual_biomass_input_100y = np.full(SIMULATION_YEARS, press_cake_biomass_input)
        annual_biomass_input_30y = np.array([
            press_cake_biomass_input if yr < production_stop_year else 0 for yr in time
        ])
        annual_biomass_input_double_ley = np.array([
            press_cake_biomass_input if yr < double_ley_start_year else press_cake_biomass_four_leys
            for yr in time
        ])

        def calculate_carbon_dynamics(annual_biomass_input):
            effective_in = annual_biomass_input * carbon_ADPC * biogas_yield * LOSS_FACTOR * biomethane_to_C2_to_PHB

            carbon_stored = np.zeros(SIMULATION_YEARS)
            carbon_released = np.zeros(SIMULATION_YEARS)

            for yr in range(SIMULATION_YEARS):
                carbon_stored[yr] = (carbon_stored[yr - 1] if yr else 0) + effective_in[yr]

            mean_release_time = 2
            sd_release_time = 1

            normalization = np.zeros(SIMULATION_YEARS)
            for start in range(SIMULATION_YEARS):
                τ = np.arange(SIMULATION_YEARS - start)
                pdf = np.exp(-0.5 * ((τ - mean_release_time) / sd_release_time) ** 2) / (sd_release_time * np.sqrt(2 * np.pi))
                normalization[start] = pdf.sum()

            for yr in range(SIMULATION_YEARS):
                for prev in range(yr + 1):
                    age = yr - prev
                    frac = (np.exp(-0.5 * ((age - mean_release_time) / sd_release_time) ** 2) /
                            (sd_release_time * np.sqrt(2 * np.pi))) / normalization[prev]
                    rel = effective_in[prev] * frac
                    rel = min(rel, carbon_stored[prev])  # safeguard
                    carbon_released[yr] += rel

            net_seq = np.maximum(carbon_stored - np.cumsum(carbon_released), 0)
            return carbon_stored / 1000, carbon_released / 1000, net_seq / 1000

        cS_100, cR_100, net_100 = calculate_carbon_dynamics(annual_biomass_input_100y)
        cS_30,  cR_30,  net_30  = calculate_carbon_dynamics(annual_biomass_input_30y)
        cS_DL,  cR_DL,  net_DL  = calculate_carbon_dynamics(annual_biomass_input_double_ley)

        plot_years = 100
        time_plot = np.arange(plot_years)

        net_carbon_df = pd.DataFrame({
            "Year": time_plot,
            "Net Carbon 100y (tons)": net_100[:plot_years],
            "Net Carbon 30y (tons)": net_30[:plot_years],
            "Net Carbon Double Ley (tons)": net_DL[:plot_years],
        })
        net_carbon_df.to_excel("enter path here", index=False)
        net_carbon_df.to_pickle("enter path here")

        colors = {
            "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759", "teal": "#76B7B2",
            "green": "#59A14F", "yellow": "#EDC948", "pink": "#FF9DA7",
            "brown": "#9C755F", "gray": "#BAB0AC"
        }

        plt.figure(figsize=(12, 6))

        # Panel 1
        plt.subplot(1, 2, 1)
        plt.plot(time_plot, cS_30[:plot_years],  label="Scenario 1 C input", color=colors["gray"],  ls='-.', lw=3)
        plt.plot(time_plot, cS_100[:plot_years], label="Scenario 2 C input", color=colors["brown"], ls='-.', lw=3)
        plt.plot(time_plot, cS_DL[:plot_years],  label="Scenario 3 C input", color=colors["yellow"], ls='-.', lw=3)

        plt.plot(time_plot, np.cumsum(cR_30)[:plot_years],  label="Scenario 1 C Release", color=colors["teal"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_100)[:plot_years], label="Scenario 2 C Release", color=colors["pink"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_DL)[:plot_years],  label="Scenario 3 C Release", color=colors["red"],  ls='--', lw=3)

        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=3)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=3)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=3)

        plt.xlabel("Year")
        plt.ylabel("Carbon Dynamics [t C ha⁻¹]")
        plt.title("Short-lived plastics – all scenarios (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        # Panel 2
        plt.subplot(1, 2, 2)
        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=5)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=5)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=5)

        plt.xlabel("Year")
        plt.ylabel("Cumulative Net C Seq [t C ha⁻¹]")
        plt.title("Net carbon sequestration – short-lived plastics (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enter path here", dpi=300)
        plt.savefig("enter path here", dpi=600)
        # plt.show()

    # -------------------------------------------------------------------------
    # CH4 → long-lived plastics
    # -------------------------------------------------------------------------
    def methane_to_plastic_LONGTERM(biomass_utilization_factor=1.0, process_yield=1.0):
        SIMULATION_YEARS = MODEL_YEARS_USE
        production_stop_year = 30
        double_ley_start_year = 30
        time = np.arange(SIMULATION_YEARS)

        press_cake_biomass_input = press_cake_biomass_input_base
        press_cake_biomass_four_leys = press_cake_biomass_four_leys_base

        annual_biomass_input_100y = np.full(SIMULATION_YEARS, press_cake_biomass_input)
        annual_biomass_input_30y = np.array([
            press_cake_biomass_input if yr < production_stop_year else 0.0 for yr in time
        ])
        annual_biomass_input_double_ley = np.array([
            press_cake_biomass_input if yr < double_ley_start_year else press_cake_biomass_four_leys
            for yr in time
        ])

        carbon_ADPC = 0.96
        biogas_yield = 0.36
        biomethane_to_C2_to_PHB = 0.50
        LOSS_FACTOR = 0.98

        def calculate_carbon_dynamics(annual_biomass_input):
            effective_in = annual_biomass_input * carbon_ADPC * biogas_yield * LOSS_FACTOR * biomethane_to_C2_to_PHB

            carbon_stored = np.zeros(SIMULATION_YEARS)
            carbon_released = np.zeros(SIMULATION_YEARS)

            for yr in range(SIMULATION_YEARS):
                carbon_stored[yr] = (carbon_stored[yr - 1] if yr else 0) + effective_in[yr]

            μ, σ = 25, 12
            normalization = np.zeros(SIMULATION_YEARS)
            for start in range(SIMULATION_YEARS):
                τ = np.arange(SIMULATION_YEARS - start)
                pdf = np.exp(-0.5 * ((τ - μ) / σ) ** 2) / (σ * np.sqrt(2 * np.pi))
                normalization[start] = pdf.sum()

            for yr in range(SIMULATION_YEARS):
                for prev in range(yr + 1):
                    age = yr - prev
                    frac = (np.exp(-0.5 * ((age - μ) / σ) ** 2) / (σ * np.sqrt(2 * np.pi))) / normalization[prev]
                    rel = effective_in[prev] * frac
                    rel = min(rel, carbon_stored[prev])
                    carbon_released[yr] += rel

            net_seq = np.maximum(carbon_stored - np.cumsum(carbon_released), 0)
            return carbon_stored / 1000, carbon_released / 1000, net_seq / 1000

        cS_100, cR_100, net_100 = calculate_carbon_dynamics(annual_biomass_input_100y)
        cS_30,  cR_30,  net_30  = calculate_carbon_dynamics(annual_biomass_input_30y)
        cS_DL,  cR_DL,  net_DL  = calculate_carbon_dynamics(annual_biomass_input_double_ley)

        plot_years = 100
        time_plot = np.arange(plot_years)

        net_carbon_df = pd.DataFrame({
            "Year": time_plot,
            "Net Carbon 100y (tons)": net_100[:plot_years],
            "Net Carbon 30y (tons)": net_30[:plot_years],
            "Net Carbon Double Ley (tons)": net_DL[:plot_years],
        })
        net_carbon_df.to_excel("enter path here", index=False)
        net_carbon_df.to_pickle("enter path here")

        colors = {
            "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759", "teal": "#76B7B2",
            "green": "#59A14F", "yellow": "#EDC948", "pink": "#FF9DA7",
            "brown": "#9C755F", "gray": "#BAB0AC"
        }

        plt.figure(figsize=(12, 6))

        # Panel 1
        plt.subplot(1, 2, 1)
        plt.plot(time_plot, cS_30[:plot_years],  label="Scenario 1 C input", color=colors["gray"],  ls='-.', lw=3)
        plt.plot(time_plot, cS_100[:plot_years], label="Scenario 2 C input", color=colors["brown"], ls='-.', lw=3)
        plt.plot(time_plot, cS_DL[:plot_years],  label="Scenario 3 C input", color=colors["yellow"], ls='-.', lw=3)

        plt.plot(time_plot, np.cumsum(cR_30)[:plot_years],  label="Scenario 1 C Release", color=colors["teal"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_100)[:plot_years], label="Scenario 2 C Release", color=colors["pink"], ls='--', lw=3)
        plt.plot(time_plot, np.cumsum(cR_DL)[:plot_years],  label="Scenario 3 C Release", color=colors["red"],  ls='--', lw=3)

        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=3)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=3)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=3)

        plt.xlabel("Year")
        plt.ylabel("Carbon Dynamics [t C ha⁻¹]")
        plt.title("Long-lived plastics – all scenarios (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        # Panel 2
        plt.subplot(1, 2, 2)
        plt.plot(time_plot, net_30[:plot_years],  label="Scenario 1 Net C", color=colors["orange"], lw=4)
        plt.plot(time_plot, net_100[:plot_years], label="Scenario 2 Net C", color=colors["green"],  lw=4)
        plt.plot(time_plot, net_DL[:plot_years],  label="Scenario 3 Net C", color=colors["blue"],   lw=4)

        plt.xlabel("Year")
        plt.ylabel("Cumulative Net C Seq [t C ha⁻¹]")
        plt.title("Net carbon sequestration – long-lived plastics (0–100 y)")
        plt.legend(fontsize=11)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enter path here", dpi=300)
        plt.savefig("enter path here", dpi=600)
        # plt.show()

    # -------------------------------------------------------------------------
    # CH4 → fuel (avoided emissions proxy)
    # -------------------------------------------------------------------------
    def CH4_to_fuel(biomass_utilization_factor=1.0, process_yield=1.0):
        MODEL_YEARS = 100
        production_stop_year = 30
        double_ley_start_year = 30
        time = np.arange(MODEL_YEARS)

        digeste_and_AD_corrections = 300
        CO2_to_C = digeste_and_AD_corrections / 3.67

        press_cake_biomass_input = 1075 - CO2_to_C
        press_cake_biomass_four_leys = 2 * press_cake_biomass_input
        carbon_fraction_in_press_cake = 1.0
        carbon_in_biomass = 1.0

        biomethane_loss = (1 - 0.06)
        ratio = 2.1 / 2.75  # avoided emissions adjustment
        direct_loss = 0.02
        loss_effect = (1 - direct_loss)

        grass_to_product_efficiency = 1.0 * biomethane_loss * ratio

        annual_biomass_input_100y = np.full(MODEL_YEARS, press_cake_biomass_input)
        annual_biomass_input_30y = np.array([
            press_cake_biomass_input if yr < production_stop_year else 0 for yr in time
        ])
        annual_biomass_input_double_ley = np.array([
            press_cake_biomass_input if yr < double_ley_start_year else press_cake_biomass_four_leys
            for yr in time
        ])

        def calculate_carbon_dynamics(annual_biomass_input):
            effective_biomass_input = annual_biomass_input * biomass_utilization_factor * process_yield
            carbon_press_cake_per_year = (
                effective_biomass_input * carbon_in_biomass * carbon_fraction_in_press_cake * grass_to_product_efficiency
            )

            carbon_stored = np.zeros(MODEL_YEARS)
            carbon_released = np.zeros(MODEL_YEARS)

            for yr in range(MODEL_YEARS):
                carbon_stored[yr] = (carbon_stored[yr - 1] if yr else 0) + carbon_press_cake_per_year[yr]

            mean_release_time = 10000  # effectively no release over the horizon
            sd_release_time = 1

            for yr in range(MODEL_YEARS):
                for prev in range(yr + 1):
                    age = yr - prev
                    frac = (1 / (sd_release_time * np.sqrt(2 * np.pi))) * \
                           np.exp(-0.5 * ((age - mean_release_time) / sd_release_time) ** 2)
                    carbon_released[yr] += carbon_press_cake_per_year[prev] * frac

            net_seq = np.maximum(carbon_stored - np.cumsum(carbon_released), 0)
            return carbon_stored / 1000, carbon_released / 1000, net_seq / 1000

        cS_100, cR_100, net_100 = calculate_carbon_dynamics(annual_biomass_input_100y)
        cS_30,  cR_30,  net_30  = calculate_carbon_dynamics(annual_biomass_input_30y)
        cS_DL,  cR_DL,  net_DL  = calculate_carbon_dynamics(annual_biomass_input_double_ley)

        # Force scenario where production stops at 30 years to zero afterward
        net_30[production_stop_year:] = 0

        net_carbon_df = pd.DataFrame({
            "Year": time,
            "Net Carbon 100y (tons)": net_100,
            "Net Carbon 30y (tons)": net_30,
            "Net Carbon Double Ley (tons)": net_DL,
        })
        net_carbon_df.to_excel("enter path here", index=False)
        net_carbon_df.to_pickle("enter path here")

        # Optional reference dataset with zeros
        net_zero_df = pd.DataFrame({
            "Year": time,
            "Net Carbon 100y (tons)": 0,
            "Net Carbon 30y (tons)": 0,
            "Net Carbon Double Ley (tons)": 0,
        })
        net_zero_df.to_excel("enter path here", index=False)
        net_zero_df.to_pickle("enter path here")

        colors = {
            "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759", "teal": "#76B7B2",
            "green": "#59A14F", "yellow": "#EDC948", "pink": "#FF9DA7",
            "brown": "#9C755F", "gray": "#BAB0AC"
        }

        plt.figure(figsize=(12, 6))

        # Panel 1
        plt.subplot(1, 2, 1)
        plt.plot(time, np.cumsum(cR_30),  label="Scenario 1 C Stored",  color=colors["orange"], lw=3)
        plt.plot(time, np.cumsum(cR_100), label="Scenario 2 C Stored",  color=colors["green"],  lw=3)
        plt.plot(time, np.cumsum(cR_DL),  label="Scenario 3 C Stored",  color=colors["blue"],   lw=3)

        plt.plot(time, net_30,  label="Scenario 1 Net C (avoided)", color=colors["teal"], ls='--', lw=3)
        plt.plot(time, net_100, label="Scenario 2 Net C (avoided)", color=colors["pink"], ls='--', lw=3)
        plt.plot(time, net_DL,  label="Scenario 3 Net C (avoided)", color=colors["red"],  ls='--', lw=3)

        plt.xlabel("Year")
        plt.ylabel("Carbon Dynamics [t C ha⁻¹]")
        plt.legend(fontsize=11)
        plt.grid(True)

        # Panel 2
        plt.subplot(1, 2, 2)
        plt.plot(time, net_30,  label="Scenario 1 Net C (avoided)", color=colors["teal"], ls='--', lw=5)
        plt.plot(time, net_100, label="Scenario 2 Net C (avoided)", color=colors["pink"], ls='--', lw=5)
        plt.plot(time, net_DL,  label="Scenario 3 Net C (avoided)", color=colors["red"],  ls='--', lw=5)

        plt.xlabel("Year")
        plt.ylabel("Cumulative Net C Seq [t C ha⁻¹]")
        plt.legend(fontsize=11)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enter path here", dpi=300)
        plt.savefig("enter path here", dpi=600)
        # plt.show()

    # -------------------------------------------------------------------------
    # End of function definitions
    # -------------------------------------------------------------------------

    # Optional: execute selected scenarios here (commented by default)
    # press_cake_to_ISOLATION()
    # press_cake_to_CELLULOSE_PACKAGING()
    # press_cake_to_BIOCHAR()
    # methane_to_plastic_SHORTTERM()
    # methane_to_plastic_LONGTERM()
    # CH4_to_fuel()


# Entry point
if __name__ == "__main__":
    # Define functions without running heavy simulations on import.
    all_parts_introduced_to_landscape_GB_AD_PY_and_process_carbon_flows()
