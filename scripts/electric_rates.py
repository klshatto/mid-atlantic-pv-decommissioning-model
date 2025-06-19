"""
Electricity Compensation Rates (electric_rates.py)

Generates state-level electricity compensation rate estimates for residential PV modeling, 
based on exponential fits to historical EIA data. A special override is applied for 
Pennsylvania using historical PTC (Price-to-Compare) values.

Outputs:
- ../outputs/state_cohort_exponential_rates.csv  
    → Forecasted start rates and escalation rates by state and cohort
- ../outputs/state_cohort_exponential_rates_with_PA_PTC.csv  
    → Same as above, but replaces PA with PTC-based values (used in econ_model.py)

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe
ENVM 670 Environmental Management Capstone  
University of Maryland Global Campus  
Dr. Sabrina Fu & Sponsor David Comis  
Spring 2025
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# --- Load EIA data ---
df = pd.read_csv('../data/EIA_State_Electricity_Rates_Residential_Over_Time.csv')
states = sorted(df['State'].unique())

# --- Load/define PA PTC rates ---
# Pennsylvania's EIA-based rates replaced with PTC-based rates from PPL Electric (2023)
pa_ptc_data = [
    (2012.9, 7.554), (2013.2, 7.237), (2013.4, 8.227), (2013.7, 8.777), (2013.9, 8.754),
    (2014.2, 8.754), (2014.4, 9.036), (2014.7, 8.814), (2014.8, 8.956), (2014.9, 9.318),
    (2015.2, 9.559), (2015.4, 9.493), (2015.9, 7.878),
    (2016.4, 7.491), (2016.9, 7.439), 
    (2017.4, 8.493), (2017.9, 7.463), 
    (2018.4, 7.449), (2018.9, 7.039),  
    (2019.4, 7.585), (2019.9, 7.632), 
    (2020.4, 7.284), (2020.9, 7.317),  
    (2021.4, 7.544), (2021.9, 9.502),  
    (2022, 8.941), (2022.4, 12.366), (2022.9, 14.162),
    (2023.4, 12.126), (2023.9, 11.028)
]
ptc_df = pd.DataFrame(pa_ptc_data, columns=['Year', 'PTC_c_per_kWh'])
ptc_df['PTC_$per_kWh'] = ptc_df['PTC_c_per_kWh'] / 100

# --- Fit exponential to PA PTC ---
years_ptc = ptc_df['Year'] - ptc_df['Year'].min()
fit_ptc = np.polyfit(years_ptc, np.log(ptc_df['PTC_$per_kWh']), 1)

def forecast_ptc(start_year, forecast_year):
    # Interpolate for start value
    base = np.interp(start_year, ptc_df['Year'], ptc_df['PTC_$per_kWh'])
    n = forecast_year - start_year
    return base * np.exp(fit_ptc[0] * n)

# --- Cohort midpoints to use ---
cohort_midpoints = [2012.5, 2017.5, 2022.5]

# --- Exponential fit for EIA (all states) ---
def exp_func(x, a, b, x0):
    return a * np.exp(b * (x - x0))

output_rows = []

for state in states:
    state_df = df[df['State'] == state].sort_values('Year')
    years = state_df['Year'].values
    rates = state_df['Price ($/kWh)'].values
    if len(years) < 3:
        print(f"Skipping {state}: Not enough data points")
        continue

    x0 = years[0]
    try:
        params, _ = curve_fit(lambda x, a, b: exp_func(x, a, b, x0), years, rates, p0=(rates[0], 0.02), maxfev=10000)
        a_fit, b_fit = params
    except RuntimeError:
        print(f"Could not fit {state}, using mean rate instead")
        a_fit = np.mean(rates)
        b_fit = 0.0

    for midpoint in cohort_midpoints:
        rate_at_midpoint = exp_func(midpoint, a_fit, b_fit, x0)
        output_rows.append({
            'State': state,
            'Cohort_Midpoint': midpoint,
            'Start_Rate': rate_at_midpoint,
            'Escalation_Rate': b_fit,
            'a_fit': a_fit,
            'b_fit': b_fit,
            'x0': x0,
            'Rate_Type': 'EIA'
        })

# --- Save EIA-based CSV (all states, including PA) ---
output_df = pd.DataFrame(output_rows)
output_df.to_csv('../outputs/state_cohort_exponential_rates.csv', index=False)
print("Saved EIA-based rates to ../outputs/state_cohort_exponential_rates.csv")

# --- Now, overwrite PA in a copy with PTC values for PA only ---
ptc_rates = [forecast_ptc(mid, mid) for mid in cohort_midpoints]
output_df_pa_ptc = output_df.copy()

for idx, mid in enumerate(cohort_midpoints):
    mask = (output_df_pa_ptc['State'] == 'PA') & (np.isclose(output_df_pa_ptc['Cohort_Midpoint'], mid))
    output_df_pa_ptc.loc[mask, 'Start_Rate'] = ptc_rates[idx]
    output_df_pa_ptc.loc[mask, 'Rate_Type'] = 'PTC'

output_df_pa_ptc.to_csv('../outputs/state_cohort_exponential_rates_with_PA_PTC.csv', index=False)
print("Saved rates with PA PTC to ../outputs/state_cohort_exponential_rates_with_PA_PTC.csv")

