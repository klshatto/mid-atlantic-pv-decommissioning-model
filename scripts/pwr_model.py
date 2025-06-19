"""
Power Decrease Model (pwr_model.py)

Forecasts the end-of-life (EoL) distribution of residential PV systems in the Mid-Atlantic 
region using a power degradation Weibull model.

- Reads in cleaned PV installation data with PV climate zones (PVCZs; Karin et al., 2019)
- Bins systems into 5-year installation cohorts
- Applies Weibull-based modeling (α, β) to estimate EoL due to power degradation
- Aggregates by PVCZ and cohort for each scenario (α)
- Outputs forecasted removals to CSV and generates multiple diagnostic plots

Expected Input:
- ../data/merged_midatlantic_residential_pv.csv
    → Required columns: 'Year Online', 'PVCZ'
    → Assumes residential system data from 2005–2024 with assigned PVCZs

Outputs:
- ../outputs/pv_eol_forecast_by_pvcz_power_decrease.csv
    → Forecasted removals by year, PVCZ, cohort, and Weibull α
- ../outputs/pwr_model_means.csv
    → Mean calendar year and mean age at EoL, by PVCZ and α
- ../outputs/plots/
    → Weibull CDF and PDF curves per PVCZ
    → Forecasted removals by year (per PVCZ and all-in-one view)

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe
ENVM 670 Environmental Management Capstone
University of Maryland Global Campus
Dr. Sabrina Fu & Sponsor David Comis
Spring 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min
import os
os.makedirs("../outputs/plots", exist_ok=True)

# Set global plot style
import seaborn as sns
sns.set(style="whitegrid", rc={"grid.color": "lightgray", "grid.linestyle": "--"})

# === 1. DATA IMPORT & CLEANING ===

base_path = "../data/"
input_file = os.path.join(base_path, "merged_midatlantic_residential_pv.csv")
pv_df = pd.read_csv(input_file)

# Ensure years are numeric, drop incomplete rows, cast to int
pv_df['Year Online'] = pd.to_numeric(pv_df['Year Online'], errors='coerce')
pv_df = pv_df.dropna(subset=['Year Online'])
pv_df['Year Online'] = pv_df['Year Online'].astype(int)

# === 2. BINNING SYSTEMS INTO INSTALLATION COHORTS ===

pv_df['Install_Cohort'] = pd.cut(
    pv_df['Year Online'],
    bins=[2005, 2010, 2015, 2020, 2025],
    labels=['2005–2009', '2010–2014', '2015–2019', '2020–2024'],
    right=False
)
install_midpoints = {
    '2005–2009': 2007,
    '2010–2014': 2012,
    '2015–2019': 2017,
    '2020–2024': 2022,
}
# Groups: Each system is assigned a 5-year install cohort and a PVCZ.

# === 3. AGGREGATE BY COHORT AND PVCZ ===

cohort_counts = (
    pv_df.groupby(['PVCZ', 'Install_Cohort'])
    .size()
    .unstack(fill_value=0)
)

# === 4. DEFINE WEIBULL PARAMETERS (FROM PLR ANALYSIS) ===

pvcz_beta = {'T6:H4': 20.2, 'T7:H4': 18.5, 'T8:H4': 17.1}   # PLR -1.1: 20.2; -1.2: 18.5; -1.3 17.1
                                                            # PLR -0.4: 55.7; -0.5: 44.5; -0.6: 37.1 

alphas = [2.4928, 5.3759]  # Sensitivity: early loss (industry) vs. regular loss (lab-based)

# === 5. FORECAST EOL DISTRIBUTIONS BY PVCZ, COHORT, ALPHA ===

years = np.arange(2025, 2061)
eol_forecasts = []

# For each PVCZ, cohort, and alpha, simulate EoL distribution
for pvcz in cohort_counts.index:
    for cohort_label in cohort_counts.columns:
        installs = cohort_counts.loc[pvcz, cohort_label]
        if installs == 0:
            continue

        # Get the midpoint install year for cohort
        start_year = int(cohort_label.split('–')[0])
        midpoint_year = start_year + 2  # approximate midpoint of 5-year span
        beta = pvcz_beta[pvcz]

        for alpha in alphas:
            # Simulate EoL year distribution from Weibull
            lifetime_cdf = weibull_min.cdf(years - midpoint_year, c=alpha, scale=beta)
            lifetime_cdf_prev = weibull_min.cdf(years - 1 - midpoint_year, c=alpha, scale=beta)
            fraction_per_year = lifetime_cdf - lifetime_cdf_prev
            eol_counts = installs * fraction_per_year

            for year, count in zip(years, eol_counts):
                eol_forecasts.append({
                    'PVCZ': pvcz,
                    'Cohort': cohort_label,
                    'Alpha': alpha,
                    'Year': year,
                    'Expected EoL Systems': count
                })

# Create a DataFrame with EoL projections and write to CSV
eol_df = pd.DataFrame(eol_forecasts)
eol_df['Install_Midpoint'] = eol_df['Cohort'].map(install_midpoints)
eol_df['System_Age'] = eol_df['Year'] - eol_df['Install_Midpoint']
eol_df.to_csv("../outputs/pv_eol_forecast_by_pvcz_power_decrease.csv", index=False)
print("Saved EoL projections to CSV.")

# === 6. AGGREGATE/SUMMARIZE AND PLOT ===

# Summarize by year, alpha, and PVCZ
summary = (
    eol_df.groupby(['PVCZ', 'Alpha'])
    .apply(lambda df: pd.Series({
        "Mean_Calendar_Year": np.average(df['Year'], weights=df['Expected EoL Systems']),
        "Mean_System_Age": np.average(df['System_Age'], weights=df['Expected EoL Systems'])
    }))
    .reset_index()
)
summary.to_csv("../outputs/pwr_model_means.csv", index=False)
print(summary)

# Forecasted EoL by PVCZ (one plot per PVCZ, both alphas overlaid)
for pvcz in eol_df['PVCZ'].unique():
    data = eol_df[eol_df['PVCZ'] == pvcz]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Year', y='Expected EoL Systems', hue='Alpha', palette='tab10', errorbar=None)
    plt.title(f"Forecasted PV End-of-Life Systems – PVCZ {pvcz} – Power Decrease Model")
    plt.ylabel("Expected EoL Systems")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title="Alpha")
    plt.tight_layout()
    plt.show()

# Weibull PDFs and CDFs by PVCZ
x_vals = np.linspace(0, 40, 500)  # years since installation

for pvcz, beta in pvcz_beta.items():
    # PDF
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        pdf_vals = weibull_min.pdf(x_vals, c=alpha, scale=beta)
        plt.plot(x_vals, pdf_vals, label=f'α = {alpha}', linewidth=2)
    plt.title(f"Weibull PDF – PVCZ {pvcz} (β = {beta})")
    plt.xlabel("Years Since Installation")
    plt.ylabel("Probability Density")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title="Shape (α)")
    plt.tight_layout()
    plt.show()

    # CDF
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        cdf_vals = weibull_min.cdf(x_vals, c=alpha, scale=beta)
        plt.plot(x_vals, cdf_vals, label=f'α = {alpha}', linewidth=2)
    plt.title(f"Weibull CDF – PVCZ {pvcz} (β = {beta})")
    plt.xlabel("Years Since Installation")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title="Shape (α)")
    plt.tight_layout()
    plt.show()

# All PDFs and CDFs in one figure (subplots)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

for idx, (pvcz, beta) in enumerate(pvcz_beta.items()):
    # CDF
    ax_cdf = axes[idx, 0]
    for alpha in alphas:
        cdf_vals = weibull_min.cdf(x_vals, c=alpha, scale=beta)
        ax_cdf.plot(x_vals, cdf_vals, label=f'α = {alpha}', linewidth=2)
    ax_cdf.set_title(f'Weibull CDF – {pvcz} (β = {beta})')
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.grid(True, linestyle='--', linewidth=0.5)
    if idx == 2:
        ax_cdf.set_xlabel("Years Since Installation")

    # PDF
    ax_pdf = axes[idx, 1]
    for alpha in alphas:
        pdf_vals = weibull_min.pdf(x_vals, c=alpha, scale=beta)
        ax_pdf.plot(x_vals, pdf_vals, label=f'α = {alpha}', linewidth=2)
    ax_pdf.set_title(f'Weibull PDF – {pvcz} (β = {beta})')
    ax_pdf.set_ylabel("Probability Density")
    ax_pdf.grid(True, linestyle='--', linewidth=0.5)
    if idx == 2:
        ax_pdf.set_xlabel("Years Since Installation")

axes[0, 0].legend(title="Shape (α)")
axes[0, 1].legend(title="Shape (α)")
plt.tight_layout()
plt.savefig("../outputs/plots/pwr_all_in_one_weibull.png")
plt.show()

# One long figure with a subplot per PVCZ (forecasted EoL)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)

for ax, pvcz in zip(axes, eol_df['PVCZ'].unique()):
    data = eol_df[eol_df['PVCZ'] == pvcz]
    sns.lineplot(data=data, x='Year', y='Expected EoL Systems', hue='Alpha', palette='tab10', ax=ax, errorbar=None)
    ax.set_title(f'Forecasted PV End-of-Life Systems – {pvcz} – Power Decrease Model')
    ax.set_ylabel("Expected EoL Systems")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(title="Alpha")

axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.savefig("../outputs/plots/pwr_all_in_one_forecast.png")
plt.show()
