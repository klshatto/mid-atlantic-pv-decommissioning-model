"""
Economic Model (econ_model.py)

Forecasts the economic end-of-life (EoL) and optimal replacement timing for residential PV 
systems in the Mid-Atlantic region, based on NPV modeling of baseline and replacement scenarios.

- Loads system installation, cost, degradation, and compensation data by cohort and PVCZ
- Computes average system size and net initial costs by PVCZ and cohort
- Simulates annual cash flows and NPV for both baseline and replacement scenarios
- Identifies break-even and optimal replacement ages for NPV-maximizing outcomes
- Outputs detailed results to CSV and generates summary plots for scenario analysis

Expected Input:
- ../outputs/pv_eol_forecast_by_pvcz_power_decrease.csv  
    → Forecasted installs by PVCZ and cohort (from power model)
- ../data/merged_midatlantic_residential_pv.csv  
    → Residential PV install dataset with PVCZ and system size
- ../data/EIA_State_Electricity_Rates_Residential_Over_Time_w_PA_PTC.csv  
    → Historical electricity rates by state and year (for Pt mapping)

Outputs:
- ../outputs/economic_lifetime_results.csv  
    → Summary of optimal replacement timing and NPV improvements
- ../outputs/full_npv_audit_by_year.csv  
    → Year-by-year NPV audit by PVCZ and cohort
- ../outputs/avg_system_size_by_pvcz_and_cohort.csv  
    → Average system size and dynamic cost estimates
- ../outputs/plots/  
    → NPV bar plots, replacement/break-even age histograms, Weibull fit, etc.

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe
ENVM 670 Environmental Management Capstone  
University of Maryland Global Campus  
Dr. Sabrina Fu & Sponsor David Comis  
Spring 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import seaborn as sns
import os
os.makedirs("../outputs/plots", exist_ok=True)

# Set global plot style
sns.set(style="whitegrid", rc={"grid.color": "lightgray", "grid.linestyle": "--"})

# === 1. CONFIGURATION AND CONSTANTS ===
# Financial assumptions and modeling parameters

def main():

    # --- Filepaths for key input datasets ---
    POWER_EOL_FILE = "../outputs/pv_eol_forecast_by_pvcz_power_decrease.csv"
    MERGED_PV_FILE = "../data/merged_midatlantic_residential_pv.csv"
 #   COUNTY_PVCZ_FILE = "../data/midatlantic_county_pvcz.csv"
    EIA_RATES = "../data/EIA_State_Electricity_Rates_Residential_Over_Time_w_PA_PTC.csv"
 #   EIA_RATES = "../data/EIA_State_Electricity_Rates_Residential_Over_Time.csv"
        # For sensitivity analysis - what if PA used full retail rates compensation?

    # --- Financial and technical constants ---
    discount_rate = 0.05
    maintenance_cost = 100
    horizon_years = 30
    breakeven_threshold = 0  # NPV improvement threshold for replacement

    # p values by PVCZ; GHI_mean (kWh/m^2/day) x performance ratio (0.79) x 365
    # GHI_mean values based on Karin et al. (2019; see merge.py) and DOE (2021) PR guidance
    p_by_pvcz = {
        "T6:H4": 1075.112,
        "T7:H4": 1153.276,
        "T8:H4": 1251.899
    }
    
    # Upgrade factor by cohort (from prep_track_sun_params.py)
    upgrade_factor_by_cohort = {
        "2010–2014": 1.512579,
        "2015–2019": 1.196517,
        "2020–2024": 1.002083,
    }

    # Performance loss rate (PLR) by PVCZ (percent per year)
    degradation_by_pvcz = {
        "T6:H4": 0.011,         # Reset to 0.011
        "T7:H4": 0.012,         # Reset to 0.012
        "T8:H4": 0.013,         # Reset to 0.013
    }

    # Rebate by cohort (from prep_track_sun_params.py)
    rebate_by_cohort = {
        "2010–2014": 6041.237025,
        "2015–2019": 501.005805,
        "2020–2024": 897.346517,
    }

    COHORT_LABEL_TO_MIDPOINT = {
        "2005–2009": 2007.0,
        "2010–2014": 2012.5,
        "2015–2019": 2017.5,
        "2020–2024": 2022.5
    }
 
    # === 2. LOAD AND PREPARE INPUT DATA ===

    # --- 2.1 Load Power Decrease EoL forecast
    eol_df = pd.read_csv(POWER_EOL_FILE)
    eol_df['Cohort Start Year'] = eol_df['Cohort'].str.extract(r'(\d{4})').astype(int)
    eol_df['Cohort Label'] = (
        eol_df['Cohort Start Year'].astype(str) + '–' + (eol_df['Cohort Start Year'] + 4).astype(str)
    )

    # Group total installs per PVCZ and Cohort
    cohort_counts = (
        eol_df.groupby(['PVCZ', 'Cohort Label'])['Expected EoL Systems']
        .sum().round(0).astype(int).unstack(fill_value=0)
    )
    cohort_labels = cohort_counts.columns.tolist()
    cohort_midpoints = {label: int(label.split("–")[0]) + 2 for label in cohort_labels}

    # --- 2.2 Load residential PV system install data
    merged_df = pd.read_csv(MERGED_PV_FILE)
    merged_df = merged_df.rename(columns={"Capacity (kWh)": "System Size (kW)"})
    merged_df = merged_df.dropna(subset=["PVCZ", "Year Online", "System Size (kW)"])
    merged_df["Year Online"] = merged_df["Year Online"].astype(int)

    # Assign each system to a 5-year cohort label
    merged_df["Cohort Start"] = (merged_df["Year Online"] // 5) * 5
    merged_df["Cohort Label"] = (
        merged_df["Cohort Start"].astype(str) + "–" + (merged_df["Cohort Start"] + 4).astype(str)
    )

    # Electricity compensation rates by state (historical [EIA; PPL Electric for PA PTC] w/ exponential best fit)
    eia_rates = pd.read_csv(EIA_RATES)
    eia_2024 = eia_rates[eia_rates["Year"] == 2024].set_index("State")["Price ($/kWh)"].to_dict()

    # Map historical rate (by State and Year) to each system
    def get_hist_rate(row):
        sub = eia_rates[(eia_rates["State"] == row["State"]) & (eia_rates["Year"] == row["Year Online"])]
        if not sub.empty:
            return sub["Price ($/kWh)"].values[0]
        else:
            return np.nan
    merged_df["hist_rate"] = merged_df.apply(get_hist_rate, axis=1)

    # Capacity-weighted mean by PVCZ and year
    def weighted_avg(group):
        return np.average(group["hist_rate"], weights=group["System Size (kW)"])
    pvcz_year_rates = merged_df.groupby(["PVCZ", "Year Online"]).apply(weighted_avg).unstack()

    def get_full_rate_series(pvcz, start_year, years=30, escalation=0.02):
        """
        For a given PVCZ and start_year, returns a list of annual rates:
        - capacity-weighted historical PVCZ rates for years ≤ 2024 (if available)
        - 2% escalation from the 2024 value for later years.
        """
        rate_list = []
        for t in range(years):
            year = int(start_year + t)
            # Use historical PVCZ average if available and year ≤ 2024
            if year in pvcz_year_rates.columns and year <= 2024:
                rate = pvcz_year_rates.loc[pvcz, year]
                rate_list.append(rate)
            elif year > 2024:
                base_rate = pvcz_year_rates.loc[pvcz, 2024]
                n = year - 2024
                rate_list.append(base_rate * ((1 + escalation) ** n))
            else:
                # If no rate available, fallback to first non-NaN (should be rare)
                rate_list.append(pvcz_year_rates.loc[pvcz].dropna().iloc[0])
        return rate_list


    # --- 2.3 Construct Pt (compensation rate) time series
    avg_K0_by_pvcz_cohort = (
        merged_df.groupby(["PVCZ", "Cohort Label"])["System Size (kW)"]
        .mean().round(2).unstack(fill_value=None)
    )

    avg_K0_df = avg_K0_by_pvcz_cohort.stack().reset_index()
    avg_K0_df.columns = ["PVCZ", "Cohort", "Avg System Size (kW)"]
    avg_K0_df["Midpoint Year"] = avg_K0_df["Cohort"].str.extract(r"(\d{4})").astype(int) + 2

    # Save to outputs
    avg_K0_df.to_csv("../outputs/avg_system_size_by_pvcz_and_cohort.csv", index=False)

    # === 3. DERIVE COST AND SIZE ESTIMATES ===

    # --- 3.1 Historical cost-per-watt data (from NREL/Statista; observed and extrapolated)
    # These are national average install prices by year, used for both initial and replacement cost modeling.
    cost_per_watt_data = {
        2010: 7.53, 2011: 6.62, 2012: 4.67, 2013: 4.09, 2014: 3.60,
        2015: 3.36, 2016: 3.16, 2017: 2.94, 2018: 2.78, 2019: 2.77,
        2020: 2.71, 2021: 3.05, 2022: 2.68, 2023: 3.39, 2024: 3.00
    }
    years = np.array(list(cost_per_watt_data.keys()))
    costs = np.array(list(cost_per_watt_data.values()))

    # --- 3.2 Fit exponential decline to cost-per-watt data
    def exp_decline(year, a, b):
        """Exponential cost-per-watt model: y = a * exp(b * (year - 2010)) + 1.5"""
        return a * np.exp(b * (year - 2010)) + 1.50

    # Fit curve to historical data with horizontal asymptote at 1.5 (avoid unrealistically low $/W)
    params, _ = curve_fit(exp_decline, years, costs, p0=(6, -0.1))

    def predict_cost_per_watt(year):
        """Predict $/W for a given year using exponential best fit."""
        return exp_decline(year, *params)

    # --- 3.3 Assign mid-year to each cohort for dynamic cost estimation
    cohort_midyears = {"2010–2014": 2012, "2015–2019": 2017, "2020–2024": 2022}
    avg_K0_df["Cohort Midyear"] = avg_K0_df["Cohort"].map(cohort_midyears)

    # --- 3.4 Compute $/W and initial system cost per cohort/PVCZ
    avg_K0_df["$/Watt (Dynamic)"] = avg_K0_df["Cohort Midyear"].apply(predict_cost_per_watt)
    avg_K0_df["Estimated Initial Cost ($)"] = (
        avg_K0_df["Avg System Size (kW)"] * 1000 * avg_K0_df["$/Watt (Dynamic)"]
    ).round(2)
    avg_K0_df["Mean Rebate ($)"] = avg_K0_df["Cohort"].map(rebate_by_cohort)
    avg_K0_df["Estimated Net Initial Cost ($)"] = (
        avg_K0_df["Estimated Initial Cost ($)"] - avg_K0_df["Mean Rebate ($)"]
    ).clip(lower=0)

    # Export
    avg_K0_df.to_csv("../outputs/avg_system_size_and_cost_by_pvcz_and_cohort.csv", index=False)

    # --- 3.5 Prepare lookup for initial net cost by PVCZ & Cohort
    initial_cost_by_pvcz_cohort = (
        avg_K0_df.set_index(["PVCZ", "Cohort"])["Estimated Net Initial Cost ($)"]
        .to_dict()
    )

    # === 4. CASH FLOW AND NPV MODELING FUNCTION ===

    def calculate_cash_flow_and_npv(
        Pt_by_year,
        K0,
        initial_cost,
        start_year,
        annual_degradation_rate,
        horizon_years=horizon_years,
        replacement_year=None,
        replacement_K0=None,
        replacement_cost=None,
        repair_cost_per_year=150,
        repair_cost_start_year=21
    ):
        """
        Compute annual net cash flows and NPV for a PV system scenario.

        Args:
            Pt: Electricity compensation rate ($/kWh)
            K0: Initial system size (kW)
            initial_cost: Net installation cost ($)
            start_year: Calendar year of installation
            annual_degradation_rate: Fractional annual degradation (PLR)
            horizon_years: Total analysis period (years)
            replacement_year: Year replacement system is installed (None if no replacement)
            replacement_K0: New system size after replacement (kW)
            replacement_cost: Replacement system cost ($)
            repair_cost_per_year: Annual repair cost post-warranty
            repair_cost_start_year: Warranty period before repair costs kick in

        Returns:
            cash_flows_df: DataFrame of annual cash flow details
            npv: Net Present Value ($)
        """

        cash_flows = []
        npv = -initial_cost  # Upfront cost as negative cash flow

        for t in range(horizon_years):
            Pt_t = Pt_by_year[t]
            year = start_year + t
            system_age = year - start_year

            # Are we using the original or replacement system?
            if replacement_year is not None and year >= replacement_year:
                # Replacement scenario (new warranty, new system size)
                effective_t = year - replacement_year
                current_K0 = replacement_K0 if replacement_K0 is not None else K0
                under_warranty = effective_t < 25  # New warranty after replacement
            else:
                # Still on original system
                effective_t = t
                current_K0 = K0
                under_warranty = system_age < repair_cost_start_year  # Original warranty

            # Power output, degraded each year
            Et = (1 - annual_degradation_rate) ** effective_t

            # Calculate annual revenue, maintenance, repair, replacement
            Gt = Pt_t * p * Et * current_K0            # Annual gross revenue
            Rt = maintenance_cost                      # Regular maintenance
            Ct = replacement_cost if replacement_year == year else 0
            repair_cost = 0 if under_warranty else repair_cost_per_year

            # Net cash flow
            net_cash_flow = Gt - Rt - Ct - repair_cost

            # Discount to present value
            discount_factor = (1 + discount_rate) ** t
            discounted_cash_flow = net_cash_flow / discount_factor

            npv += discounted_cash_flow

            # Collect cash flow details for this year
            cash_flows.append({
                "Year": year,
                "Revenue ($)": round(Gt, 2),
                "Maintenance ($)": Rt,
                "Repair ($)": repair_cost,
                "Replacement ($)": Ct,
                "Net Cash Flow ($)": round(net_cash_flow, 2)
            })

        return pd.DataFrame(cash_flows), round(npv, 2)

    # === 5. ECONOMIC REPLACEMENT SCENARIO MODELING ===
    # For each PVCZ and cohort, compute the NPV for baseline and each replacement scenario.
    # Track break-even year (first NPV improvement) and best replacement year (max NPV).
    economic_lifetime_results = []
    all_npv_records = []

    # Replacement age logic parameters
    min_replacement_age = 5      # Earliest replacement considered (years after install)
    max_replacement_age = 30     # Latest replacement considered

    # Loop through all combinations of PVCZ and cohort (where systems exist)
    for pvcz in cohort_counts.index:
        cohort_midpoints = {
            "2010–2014": 2012.5,
            "2015–2019": 2017.5,
            "2020–2024": 2022.5
        }
        for cohort_label in cohort_labels:
            start_year = int(COHORT_LABEL_TO_MIDPOINT[cohort_label])
            Pt_by_year = get_full_rate_series(pvcz, start_year, years=horizon_years, escalation=0.02)

        degradation_rate = degradation_by_pvcz.get(pvcz)
        p = p_by_pvcz[pvcz]

        for cohort_label in cohort_labels:
            installs = cohort_counts.loc[pvcz, cohort_label]
            if installs == 0:
                continue  # Skip if no systems installed

            # Look up average system size and cost
            if (
                pvcz not in avg_K0_by_pvcz_cohort.index or
                cohort_label not in avg_K0_by_pvcz_cohort.columns
            ):
                continue  # Skip if no data

            K0 = avg_K0_by_pvcz_cohort.loc[pvcz, cohort_label]
            initial_cost = initial_cost_by_pvcz_cohort.get((pvcz, cohort_label), None)
            start_year = cohort_midpoints.get(cohort_label, None)

            if pd.isna(K0) or pd.isna(initial_cost) or start_year is None:
                continue  # Skip if missing key data

            # Get the cohort-specific upgrade factor (how much larger new system is expected to be)
            this_upgrade_factor = upgrade_factor_by_cohort.get(cohort_label, 1.3)

            # Compute base NPV (no replacement scenario)
            _, base_npv = calculate_cash_flow_and_npv(
                Pt_by_year, K0, initial_cost, start_year=start_year, annual_degradation_rate=degradation_rate
            )

            # Track best replacement age, break-even age, and related NPVs
            best_effective_npv = None
            best_year = None
            best_raw_npv = None
            break_even_year = None
            break_even_npv = None
            max_npv_with_replacement = None
            max_year = None

            # Try all possible replacement ages (5 to 30 years after install)
            for offset in range(min_replacement_age, min(horizon_years, max_replacement_age)):
                repl_year = start_year + offset

                # Calculate dynamic upgrade for this cohort
                this_upgrade_factor = upgrade_factor_by_cohort.get(cohort_label, 1.3)

                # Predict future cost-per-watt for this replacement year
                repl_cost_per_watt = predict_cost_per_watt(repl_year)
                replacement_K0 = K0 * this_upgrade_factor
                replacement_cost = replacement_K0 * 1000 * repl_cost_per_watt

                # Compute NPV if replaced at this year
                _, npv_with_replacement = calculate_cash_flow_and_npv(
                    Pt_by_year,
                    K0,
                    initial_cost,
                    start_year=start_year,
                    annual_degradation_rate=degradation_rate,
                    replacement_year=repl_year,
                    replacement_K0=replacement_K0,
                    replacement_cost=replacement_cost,
                    horizon_years=horizon_years
                )

                # Break-even: first year where NPV improves above baseline
                if break_even_year is None and (npv_with_replacement - base_npv) > breakeven_threshold:
                    break_even_year = repl_year
                    break_even_npv = npv_with_replacement
                    break_even_age = repl_year - start_year

                # Effective NPV: apply additional decay factor (heuristic decision) for very late replacements
                # Note: This made minimal impact on results for the Mid-Atlantic, not effecting reported values. 
                decay_rate = 0.99 if offset <= 25 else 0.97
                effective_npv = npv_with_replacement * (decay_rate ** offset)

                # Track the best NPV (raw and effective)
                if best_effective_npv is None or effective_npv > best_effective_npv:
                    best_effective_npv = effective_npv
                    best_raw_npv = npv_with_replacement
                    best_year = repl_year

                if (max_npv_with_replacement is None) or (npv_with_replacement > max_npv_with_replacement):
                    max_npv_with_replacement = npv_with_replacement
                    max_year = repl_year
                
                # Collect year-by-year replacement NPV results for auditing
                all_npv_records.append({
                    "PVCZ": pvcz,
                    "Cohort": cohort_label,
                    "Start Year": start_year,
                    "Replacement Year": repl_year,
                    "System Age": repl_year - start_year,
                    "Base NPV": round(base_npv, 2),
                    "Replacement NPV": round(npv_with_replacement, 2),
                    "Difference (Replacement - Base)": round(npv_with_replacement - base_npv, 2)
                })

            # Store results for this PVCZ/cohort
            economic_lifetime_results.append({
                "PVCZ": pvcz,
                "Cohort": cohort_label,
                "Start Year": start_year,
                "Base NPV": round(base_npv, 2),
                "Best Replacement Year": max_year,
                "NPV (Best Replacement)": round(max_npv_with_replacement, 2) if max_npv_with_replacement else None,
                "First Break-even Year": break_even_year,
                "NPV (Break-even)": round(break_even_npv, 2) if break_even_npv else None,
                "Break-even Age": break_even_age if break_even_year else None,
                "Improved": (
                    pd.notna(base_npv) and
                    pd.notna(best_raw_npv) and
                    best_raw_npv > base_npv + breakeven_threshold
                )
            })

    # Convert to DataFrame and add summary columns
    economic_lifetime_df = pd.DataFrame(economic_lifetime_results)
    economic_lifetime_df['Best Replacement Year'] = economic_lifetime_df["Best Replacement Year"] + 1
    economic_lifetime_df["Best Replacement Age"] = (
        economic_lifetime_df["Best Replacement Year"] - economic_lifetime_df["Start Year"]
    )
    economic_lifetime_df["Break-even Age"] = economic_lifetime_df.apply(
        lambda row: row["First Break-even Year"] - row["Start Year"] if pd.notna(row["First Break-even Year"]) else None,
        axis=1
    )

    print(economic_lifetime_df)

    # === 6. EXPORT AND SUMMARIZE RESULTS ===

    # --- 6.1 Export detailed results to CSV
    economic_lifetime_df.to_csv("../outputs/economic_lifetime_results.csv", index=False)
    print("Results written to economic_lifetime_results.csv")

    full_npv_df = pd.DataFrame(all_npv_records)
    full_npv_df.to_csv("../outputs/full_npv_audit_by_year.csv", index=False)
    print("Exported detailed year-by-year NPV audit to ../outputs/full_npv_audit_by_year.csv")

    # --- 6.2 Add sortable cohort and group labels for downstream analysis
    economic_lifetime_df["Cohort Start"] = economic_lifetime_df["Cohort"].str.extract(r"(\d{4})").astype(int)
    economic_lifetime_df["Group"] = economic_lifetime_df["PVCZ"] + " (" + economic_lifetime_df["Cohort"] + ")"
    sorted_df = economic_lifetime_df.sort_values(by=["Cohort Start", "PVCZ"])

    # --- 6.3 Summary Table by PVCZ and Cohort
    summary = economic_lifetime_df.groupby(["PVCZ", "Cohort"]).agg({
        "Base NPV": "mean",
        "NPV (Best Replacement)": "mean",
        "Best Replacement Age": "mean",
        "Break-even Age": "mean",
        "Improved": "sum"
    }).round(2).reset_index()

    print("\nSummary Table by PVCZ and Cohort:")
    print(summary)

    # --- 6.4 Percent of Improved Systems by PVCZ
    improved_pct = (
        economic_lifetime_df.groupby("PVCZ")["Improved"]
        .mean().mul(100).round(1).reset_index(name="Improved %")
    )

    print("\nImproved Systems (%) by PVCZ:")
    print(improved_pct)

    # === 7. VISUALIZE ECONOMIC MODEL RESULTS ===

    # --- 7.1 Bar Plot: Base NPV vs Best Replacement NPV
    plt.figure(figsize=(14, 6))
    x = range(len(sorted_df))
    plt.bar(x, sorted_df["Base NPV"], width=0.4, label="Base NPV", align='center')
    plt.bar([i + 0.4 for i in x], sorted_df["NPV (Best Replacement)"], width=0.4, label="Best Replacement NPV", align='center')
    plt.xticks([i + 0.2 for i in x], sorted_df["Group"], rotation=45, ha='right')
    plt.ylabel("Net Present Value ($)")
    plt.title("Base NPV vs Best Replacement NPV by PVCZ and Cohort")
    plt.legend()
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    plt.tight_layout()
    plt.savefig("../outputs/plots/npv_comparison_plot.png")
    plt.show()

    # --- 7.2 Replacement Age Boxplot by PVCZ
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=economic_lifetime_df, x="PVCZ", y="Best Replacement Age")
    plt.title("Distribution of Best Replacement Ages by PVCZ")
    plt.ylabel("Best Replacement Age (years)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../outputs/plots/best_replacement_age_by_pvcz.png")
    plt.show()

    # --- 7.3 Break-even Age Histogram (stacked by PVCZ)
    filtered = economic_lifetime_df[economic_lifetime_df["Break-even Age"].notna()]
    if not filtered.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=filtered, x="Break-even Age", bins=15, hue="PVCZ", multiple="stack")
        plt.title("Histogram of Break-even Ages")
        plt.xlabel("Break-even Age (years)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("../outputs/plots/break_even_age_histogram.png")
        plt.show()
    else:
        print("No valid break-even age data to plot.")

    # --- 7.4 Percent of Improved Systems by PVCZ
    plt.figure(figsize=(8, 5))
    sns.barplot(data=improved_pct, x="PVCZ", y="Improved %")
    plt.title("Percent of Systems with NPV Improvement from Replacement")
    plt.ylabel("% Improved Systems")
    plt.ylim(0, 100)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("../outputs/plots/improved_systems_percent_by_pvcz.png")
    plt.show()

    # --- 7.5 Weibull Fit to Best Replacement Ages
    ages = economic_lifetime_df["Best Replacement Age"].dropna().values
    ages = ages[ages > 0]
    if len(ages) > 0:
        shape, loc, scale = weibull_min.fit(ages, floc=0)
        print(f"Fitted Weibull: alpha (shape) = {shape:.4f}, beta (scale) = {scale:.4f}")

        # Plot PDF of the fit vs histogram
        x_vals = np.linspace(min(ages), max(ages), 100)
        plt.figure(figsize=(10, 5))
        plt.hist(ages, bins=15, density=True, alpha=0.5, label="Empirical Histogram")
        plt.plot(x_vals, weibull_min.pdf(x_vals, shape, loc, scale), 'r-', label="Weibull PDF fit", linewidth=2)
        plt.title("Weibull Fit to Best Replacement Ages")
        plt.xlabel("Best Replacement Age (years)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../outputs/plots/weibull_fit_replacement_age.png")
        plt.show()
    else:
        print("No valid replacement ages for Weibull fitting.")

    # --- 7.6 Empirical CDF of Economic Model Break-evens (Step Function)
    results = economic_lifetime_df
    break_even_ages = results['Break-even Age'].dropna().astype(int).values
    if len(break_even_ages) > 0:
        ages_sorted = np.sort(break_even_ages)
        yvals = np.arange(1, len(ages_sorted) + 1) / len(ages_sorted)
        plt.figure(figsize=(8, 5))
        plt.step(ages_sorted, yvals, where='post', linewidth=2)
        plt.xlabel('Break-even Age (years)')
        plt.ylabel('Cumulative Probability')
        plt.title('Empirical CDF of Economic Model Break-evens (Step Function)')
        plt.ylim(0, 1.05)
        plt.xlim(15, max(ages_sorted) + 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("../outputs/plots/economic_model_break_even_cdf.png")
        plt.show()

    # --- 7.7 Historical Cost-per-Watt Curve and Best-Fit Visualization
    plot_years = np.arange(2010, 2051)
    fitted_costs = [predict_cost_per_watt(yr) for yr in plot_years]

    plt.figure(figsize=(10, 6))
    plt.scatter(years, costs, color='blue', label='Observed $/W (NREL/Statista)')
    plt.plot(plot_years, fitted_costs, color='red', linestyle='--', label='Exp. fit with $1.50/W floor')
    plt.axhline(1.5, color='black', linestyle=':', linewidth=1, label='$1.50/W Asymptote')
    plt.axvline(2024, color='gray', linestyle=':', linewidth=1)
    plt.text(2024.3, min(costs)+0.25, "End of Observed Data", rotation=90, color='gray')
    plt.title("Historical & Projected U.S. Residential PV Cost-per-Watt (level off to $1.50/W)")
    plt.xlabel("Year")
    plt.ylabel("Cost per Watt ($/W)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../outputs/plots/cost_curve_fit_asymptote.png")
    plt.show()

    print(f"Predicted cost-per-watt in 2030: ${predict_cost_per_watt(2030): .2f}/W")

# === 8. MAIN FUNCTION ===
if __name__ == "__main__":
    main()
