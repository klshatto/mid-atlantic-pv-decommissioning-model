"""
Tracking the Sun Data Set (prep_track_sun_params.py)

Processes the Tracking the Sun (TTS) dataset to extract parameters for residential PV 
economic modeling in the Mid-Atlantic region.

- Filters for plausible residential installations (2000–2024, >0 kW)
- Defines three cohorts (2010–2014, 2015–2019, 2020–2024)
- Calculates mean rebate/grant by cohort (used in net cost modeling)
- Calculates median system size by cohort and for 2023
- Computes upgrade factors (2023 median / cohort median)

Expected Input:
- ../data/TTS_MidAtlantic_RES.csv  
    → Cleaned residential Mid-Atlantic installs from Tracking the Sun

Outputs (printed to console):
- Mean rebate/grant by cohort
- Median system size by year and cohort
- Upgrade factor by cohort
- (*Manual copy required for integration into modeling scripts*)

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe

ENVM 670 Environmental Management Capstone
University of Maryland Global Campus
Dr. Sabrina Fu & Sponsor David Comis
Spring 2025
"""

import pandas as pd

# === 1. LOAD FILTERED DATA ===

df = pd.read_csv("../data/TTS_MidAtlantic_RES.csv")

# Exclude clear outliers and focus on plausible installs
df = df[(df['PV_system_size_DC'] > 0) & (df['installation_year'] >= 2000) & (df['installation_year'] <= 2024)]

# === 2. DEFINE COHORTS ===

cohort_bins = [2010, 2015, 2020, 2025]
cohort_labels = ['2010–2014', '2015–2019', '2020–2024']
df['cohort'] = pd.cut(df['installation_year'], bins=cohort_bins, right=False, labels=cohort_labels)

# === 3. MEAN REBATE/GRANT BY COHORT ===

rebate_by_cohort = df.groupby('cohort')['rebate_or_grant'].mean().fillna(0)
print("=== Mean rebate/grant by cohort ($): ===")
print(rebate_by_cohort)
print()

# === 4. MEDIAN SYSTEM SIZE BY YEAR & COHORT (For Upgrade Factor) ===

medians_by_year = df.groupby('installation_year')['PV_system_size_DC'].median().sort_index()
print("=== Median system size (kW) by installation year: ===")
print(medians_by_year)
print()

df_cohorts = df.dropna(subset=['cohort'])
cohort_medians = df_cohorts.groupby('cohort')['PV_system_size_DC'].median()
print("=== Median system size (kW) by cohort: ===")
print(cohort_medians)
print()

# === 5. UPGRADE FACTOR (Replacement size / original cohort size) ===

median_2023 = medians_by_year.get(2023)
print(f"=== Median system size for 2023: {median_2023} kW ===")

if median_2023 and not cohort_medians.isnull().any():
    upgrade_factors = median_2023 / cohort_medians
    print("=== Upgrade factor (2023 size / cohort size): ===")
    print(upgrade_factors)
else:
    print("Upgrade factor calculation failed due to missing 2023 data or cohort medians.")

print("\n*** Manually copy these values into your modeling scripts as needed. ***")