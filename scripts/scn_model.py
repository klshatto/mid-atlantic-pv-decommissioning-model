"""
Scenario Model (scn_model.py)

Forecasts scenario-weighted end-of-life (EoL) outcomes for residential PV systems in the 
Mid-Atlantic by blending Weibull-modeled degradation, failure, and economic behavior.

Steps:
- Uses fitted Weibull parameters for three model components (damage, power, economics)
- Blends component distributions using weighted scenarios for each PVCZ
- Applies forecasts to historical residential PV install data by cohort and PVCZ
- Calculates expected annual EoL removals and associated system ages
- Outputs CSVs and visualizations for forecasting, model components, and scenario comparisons

Expected Inputs:
- ../data/merged_midatlantic_residential_pv.csv  
    → Residential PV install dataset with PVCZ and install year columns

Outputs:
- ../outputs/scenario_summary.csv  
    → Annual EoL projections by PVCZ and scenario
- ../outputs/scenario_means.csv  
    → Scenario-weighted mean calendar year and mean system age for EoL
- ../outputs/tan_style_summary.csv  
    → Comparison of Weibull parameters and mean lifetimes by scenario and component
- ../outputs/plots/  
    → Visualizations of PDFs/CDFs and annual scenario forecasts

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe
ENVM 670 Environmental Management Capstone  
University of Maryland Global Campus  
Dr. Sabrina Fu & Sponsor David Comis  
Spring 2025
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.special import gamma
import os
os.makedirs("../outputs/plots", exist_ok=True)

# Set global plot style
sns.set(style="whitegrid", rc={"grid.color": "lightgray", "grid.linestyle": "--"})

def main():
    # === 1. DEFINE WEIBULL FUNCTIONS AND MODEL PARAMETERS ===

    def weibull_cdf(t, alpha, beta):
        """Cumulative distribution function for Weibull."""
        return 1 - np.exp(-(t / beta) ** alpha)

    def weibull_pdf(t, alpha, beta):
        """Probability density function for Weibull."""
        return (alpha / beta) * (t / beta) ** (alpha - 1) * np.exp(-(t / beta) ** alpha)

    # Forecast time range (years since installation, for CDF/PDF)
    years = np.arange(0, 40)

    # Component parameters
    alpha_dmg, beta_dmg = 2.4928, 6.9085        # Damage/Technical (from IRENA/IEA-PVPS)
    alpha_power = 2.4928                        # Power Decrease (early loss scenario)
    # Fitted Economic Model values from project analysis
    alpha_econ, beta_econ = 8.4819, 25.8207     # Economic Motivation (fitted from econ_model.py) 8.4819, 25.8207
                                                # For PA PTC sensitivity analysis (i.e., PA is full-retail rate), alpha_econ, beta_econ = 11.8912, 23.6160
                                                # For PLR -0.4%, alpha_econ, beta_econ = 10.3488, 28.1861
                                                # For PLR -0.5%, alpha_econ, beta_econ = 9.6854, 28.0137
                                                # For PLR -0.4%, -0.5%, -0.6% by PVCZ, alpha_econ, beta_econ = 9.5979, 28.0369

    # Scenario weights for 3 policy/market contexts
    scenario_weights = {
        "Scenario 1": {"dmg": 0.07, "power": 0.23, "econ": 0.70},    # High economic motivation
        "Scenario 2": {"dmg": 0.07, "power": 0.465, "econ": 0.465},  # Balanced
        "Scenario 3": {"dmg": 0.07, "power": 0.70, "econ": 0.23}     # High power-driven
    }

    # PVCZ-specific beta values for Power Decrease
    pvcz_params = {
        'T6:H4': 20.2,                          # PLR -1.1: 20.2; -0.4: 55.7; -0.5: 44.5; -0.6: 37.1
        'T7:H4': 18.5,                          # PLR -1.2: 18.5
        'T8:H4': 17.1                           # PLR -1.3: 17.1
    }

    # Storage for scenario forecast and mean lifetime
    mean_lifetimes = []
    forecast_data = {pvcz: {} for pvcz in pvcz_params}

    # === 2. COMPONENT CDF/PDF PLOTTING AND SCENARIO BLENDING ===
    # Visualize individual component and blended scenario curves by PVCZ
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

    for row_idx, (pvcz, beta_power) in enumerate(pvcz_params.items()):
        # Component CDFs and PDFs for this PVCZ
        cdf_dmg = weibull_cdf(years, alpha_dmg, beta_dmg)
        pdf_dmg = weibull_pdf(years, alpha_dmg, beta_dmg)
        cdf_power = weibull_cdf(years, alpha_power, beta_power)
        pdf_power = weibull_pdf(years, alpha_power, beta_power)
        cdf_econ = weibull_cdf(years, alpha_econ, beta_econ)
        pdf_econ = weibull_pdf(years, alpha_econ, beta_econ)

        # Plot individual components (dotted)
        ax_cdf = axes[row_idx, 0]
        ax_pdf = axes[row_idx, 1]
        ax_cdf.plot(years, cdf_dmg, 'k--', alpha=0.5, label="Damage/Technical") # Not the same alpha as Weibull!
        ax_cdf.plot(years, cdf_power, 'b--', alpha=0.5, label="Power Decrease")
        ax_cdf.plot(years, cdf_econ, 'g--', alpha=0.5, label="Economic Motivation")
        ax_pdf.plot(years, pdf_dmg, 'k--', alpha=0.5, label="Damage/Technical")
        ax_pdf.plot(years, pdf_power, 'b--', alpha=0.5, label="Power Decrease")
        ax_pdf.plot(years, pdf_econ, 'g--', alpha=0.5, label="Economic Motivation")

        # Blend scenario curves and plot
        for scenario_name, weights in scenario_weights.items():
            w_dmg, w_power, w_econ = weights["dmg"], weights["power"], weights["econ"]
            cdf_blended = w_dmg * cdf_dmg + w_power * cdf_power + w_econ * cdf_econ
            pdf_blended = w_dmg * pdf_dmg + w_power * pdf_power + w_econ * pdf_econ

            ax_cdf.plot(years, cdf_blended, label=f"{scenario_name}")
            ax_pdf.plot(years, pdf_blended, label=f"{scenario_name}")

            forecast_data[pvcz][scenario_name] = {
                "years": years,
                "cdf": cdf_blended,
                "pdf": pdf_blended
            }

            # Mean = sum of survival function = area under (1 - CDF)
            survival = 1 - cdf_blended
            mean_lifetime = np.sum(survival)

            mean_lifetimes.append({
                "Scenario": scenario_name,
                "PVCZ": pvcz,
                "Mean Lifetime (yrs)": round(mean_lifetime, 2)
            })

        ax_cdf.set_title(f"Weibull CDF – {pvcz}")
        ax_pdf.set_title(f"Weibull PDF – {pvcz}")
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_pdf.set_ylabel("Probability Density")
        if row_idx == 2:
            ax_cdf.set_xlabel("Years Since Installation")
            ax_pdf.set_xlabel("Years Since Installation")
        ax_cdf.grid(True, linestyle='--', linewidth=0.5)
        ax_pdf.grid(True, linestyle='--', linewidth=0.5)

    axes[0, 0].legend(title="Model Component / Scenario")
    axes[0, 1].legend(title="Model Component / Scenario")
    plt.tight_layout()
    plt.savefig("../outputs/plots/full_scenario_CDF_PDF.png")
    plt.show()

    # === 3. LOAD HISTORICAL INSTALLS AND FORECAST ANNUAL EOL BY SCENARIO ===

    # Load merged PV install dataset (ensure consistent with your other scripts)
    merged_file_path = "../data/merged_midatlantic_residential_pv.csv"
    merged_df = pd.read_csv(merged_file_path)

    # Clean install years and define 5-year install cohorts
    merged_df['Year Online'] = pd.to_numeric(merged_df['Year Online'], errors='coerce')
    filtered_df = merged_df.dropna(subset=['Year Online'])
    filtered_df['Year Online'] = filtered_df['Year Online'].astype(int)

    filtered_df['Install_Cohort'] = pd.cut(
        filtered_df['Year Online'],
        bins=[2005, 2010, 2015, 2020, 2025],
        labels=['2005–2009', '2010–2014', '2015–2019', '2020–2024'],
        right=False
    )

    # Group system counts by PVCZ and cohort
    cohort_counts = (
        filtered_df.groupby(['PVCZ', 'Install_Cohort'])
        .size()
        .unstack(fill_value=0)
    )

    # Set forecast range for future years
    forecast_years = np.arange(2025, 2061)
    scenario_eol_forecasts = []

    # Forecast for each PVCZ, cohort, and scenario
    for pvcz in cohort_counts.index:
        for cohort_label in cohort_counts.columns:
            installs = cohort_counts.loc[pvcz, cohort_label]
            if installs == 0:
                continue

            # Approximate cohort midpoint for lifetime modeling
            start_year = int(cohort_label.split('–')[0])
            midpoint_year = start_year + 2

            # PVCZ-specific power beta, component CDFs for this cohort
            beta_power = pvcz_params[pvcz]
            cdf_dmg = weibull_cdf(forecast_years - midpoint_year, alpha_dmg, beta_dmg)
            cdf_power = weibull_cdf(forecast_years - midpoint_year, alpha_power, beta_power)
            cdf_econ = weibull_cdf(forecast_years - midpoint_year, alpha_econ, beta_econ)

            for scenario_name, weights in scenario_weights.items():
                w_dmg, w_power, w_econ = weights["dmg"], weights["power"], weights["econ"]

                # Blend scenario CDF
                cdf_blended = w_dmg * cdf_dmg + w_power * cdf_power + w_econ * cdf_econ
                cdf_blended_prev = np.roll(cdf_blended, 1)
                cdf_blended_prev[0] = 0  # first year, no prior
                age = forecast_years - midpoint_year
                # Exclude removals before age 3 (consistent with data, avoids spurious early failures)
                fraction_per_year = np.where(age < 3, 0, cdf_blended - cdf_blended_prev)
                eol_counts = installs * fraction_per_year

                for year, count in zip(forecast_years, eol_counts):
                    scenario_eol_forecasts.append({
                        'PVCZ': pvcz,
                        'Cohort': cohort_label,
                        'Scenario': scenario_name,
                        'Year': year,
                        'Expected EoL Systems': count,
                        'System_Age': year - midpoint_year,
                    })

    # === 4. SUMMARIZE, SMOOTH, AND EXPORT RESULTS ===

    # Convert to DataFrame for downstream summary and plotting
    scenario_eol_df = pd.DataFrame(scenario_eol_forecasts)

    # Summarize by year, scenario, and PVCZ
    scenario_summary = scenario_eol_df.groupby(['Year', 'Scenario', 'PVCZ'])['Expected EoL Systems'].sum().reset_index()

    # Apply a 3-year centered rolling average (for smoother forecast curves)
    scenario_summary['Smoothed EoL Systems'] = (
        scenario_summary
        .groupby(['Scenario', 'PVCZ'])['Expected EoL Systems']
        .transform(lambda x: x.rolling(window=3, center=True, min_periods=1).mean())
    )

    # Compute mean lifetime stats: mean system age and mean calendar year for removals
    mean_stats = (
        scenario_eol_df.groupby(["Scenario", "PVCZ"])
        .apply(lambda df: pd.Series({
            "Mean Calendar Year": np.average(df["Year"], weights=df["Expected EoL Systems"]),
            "Mean System Age (yrs)": np.average(df["System_Age"], weights=df["Expected EoL Systems"])
        }))
        .reset_index()
    )
    mean_stats["Model"] = mean_stats["Scenario"]
    mean_stats = mean_stats[["Model", "PVCZ", "Mean System Age (yrs)", "Mean Calendar Year"]]

    # Export and print results for reporting
    mean_stats.to_csv("../outputs/scenario_means.csv", index=False)
    print(mean_stats)
    scenario_summary.to_csv("../outputs/scenario_summary.csv", index=False)
    print("Exported scenario_summary.csv and scenario_means.csv")

    # Function to compute Weibull mean
    def weibull_mean(alpha, beta):
        return beta * gamma(1 + 1/alpha)
    
    def sample_from_cdf(cdf, years, n_samples=10000):
        # Inverse transform sampling: for each random uniform value, find corresponding age
        randoms = np.random.uniform(0, 1, n_samples)
        sample_ages = np.interp(randoms, cdf, years)
        return sample_ages
    
    rows = []
    for scenario_name, weights in scenario_weights.items():
        for pvcz, beta_power in pvcz_params.items():
            # Scenario fitting
            # Compute blended CDF as in your plotting code
            cdf_dmg = weibull_cdf(years, alpha_dmg, beta_dmg)
            cdf_power = weibull_cdf(years, alpha_power, beta_power)
            cdf_econ = weibull_cdf(years, alpha_econ, beta_econ)
            w_dmg, w_power, w_econ = weights["dmg"], weights["power"], weights["econ"]
            cdf_blended = w_dmg * cdf_dmg + w_power * cdf_power + w_econ * cdf_econ
        
            # Sample synthetic failure ages
            ages = sample_from_cdf(cdf_blended, years)
        
            # Fit Weibull (fix location at 0 to match your modeling)
            shape, loc, scale = weibull_min.fit(ages, floc=0)

            # Damage/Technical
            rows.append({
                "Scenario": scenario_name,
                "PVCZ": pvcz,
                "Component": "Damage/Technical",
                "Weight": weights["dmg"],
                "Mean Lifetime (yrs)": round(weibull_mean(alpha_dmg, beta_dmg), 2),
                "Weibull Scale beta (yrs)": beta_dmg,
                "Weibull Shape alpha": alpha_dmg,
            })
            # Power Decrease
            rows.append({
                "Scenario": scenario_name,
                "PVCZ": pvcz,
                "Component": "Power Decrease",
                "Weight": weights["power"],
                "Mean Lifetime (yrs)": round(weibull_mean(alpha_power, beta_power), 2),
                "Weibull Scale beta (yrs)": pvcz_params[pvcz],
                "Weibull Shape alpha": alpha_power,
            })
            # Economic Motivation
            rows.append({
                "Scenario": scenario_name,
                "PVCZ": pvcz,
                "Component": "Economic Motivation",
                "Weight": weights["econ"],
                "Mean Lifetime (yrs)": round(weibull_mean(alpha_econ, beta_econ), 2),
                "Weibull Scale beta (yrs)": beta_econ,
                "Weibull Shape alpha": alpha_econ,
            })
            # Blended scenario mean (summary row)
            scenario_mean_row = mean_stats[(mean_stats["Model"] == scenario_name) & (mean_stats["PVCZ"] == pvcz)]
            mean_life = scenario_mean_row["Mean System Age (yrs)"].values[0]
            rows.append({
                "Scenario": scenario_name,
                "PVCZ": pvcz,
                "Component": "Blended (Weighted)",
                "Weight": "",
                "Mean Lifetime (yrs)": round(mean_life, 2),
                "Weibull Scale beta (yrs)": round(scale, 4),
                "Weibull Shape alpha": round(shape, 4),
            })

    # Convert to DataFrame and export
    tan_style_df = pd.DataFrame(rows)
    print(tan_style_df)
    tan_style_df.to_csv("../outputs/tan_style_summary.csv", index=False)
    print("Exported summary table to ../outputs/tan_style_summary.csv")

    # === 5. PLOTS: SCENARIO FORECASTS ===

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    for ax, pvcz in zip(axes, scenario_summary['PVCZ'].unique()):
        data = scenario_summary[scenario_summary['PVCZ'] == pvcz]
        data = data[data['Year'] >= 2027]  # Focus plots on period after initial ramp-up
        sns.lineplot(data=data, x='Year', y='Smoothed EoL Systems', hue='Scenario', palette='tab10', ax=ax)
        ax.set_title(f'Forecasted PV End-of-Life Systems – {pvcz} – Weighted Scenarios')
        ax.set_ylabel("Expected EoL Systems")
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(title="Scenario")
    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    plt.savefig("../outputs/plots/full_scenario_forecast.png")
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    for ax, pvcz in zip(axes, scenario_summary['PVCZ'].unique()):
        data = scenario_summary[scenario_summary['PVCZ'] == pvcz]
        data = data[data['Year'] >= 2027]
        for scenario in ["Scenario 1", "Scenario 2", "Scenario 3"]:
            scenario_data = data[data["Scenario"] == scenario]
            sns.lineplot(
                data=scenario_data,
                x="Year",
                y="Smoothed EoL Systems",
                label=scenario,
                ax=ax,
              #  linestyle=scenario_linestyles[scenario],
                linewidth=2.5,
               # color=scenario_colors[scenario]
            )
        ax.set_title(f'Forecasted PV End-of-Life Systems – {pvcz} – Weighted Scenarios', fontsize=20)
        ax.set_ylabel("Expected EoL Systems", fontsize=15)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(title=None, fontsize=13)
        ax.tick_params(axis='both', labelsize=13)
    axes[-1].set_xlabel("Year", fontsize=15)
    plt.tight_layout()
    plt.savefig("../outputs/plots/full_scenario_forecast_big.png")
    plt.show()

if __name__ == "__main__":
    main()
