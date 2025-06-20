# Mid-Atlantic Residential Solar PV Decommissioning Forecast

## Project Overview

This project forecasts the expected end-of-life (EoL) timing for residential silicon-based photovoltaic (PV) systems in the U.S. Mid-Atlantic region using  multi-model approach. It simulates when PV systems are likely to be decommissioned due to three main drivers:

1. **Performance degradation (Power Decrease Model)**
2. **Damage and technical failure (Damage Model)**
3. **Economic motivation for early replacement (Economic Model)**

Each model outputs time series forecasts by PV Climate Zone (PVCZ) and system cohort (based on installation year). These are integrated into weighted scenario models to evaluate regional trends and identify policy or infrastructure implications.

## Directory Structure

- `data/`: Raw input data files (installations, rates, PVCZ mappings)
- `scripts/`: All analysis and data prep scripts
- `outputs/`: Forecast outputs, plots, and summary tables
- `outputs/plots/`: Visualizations generated by each model

## Scripts and Their Purpose

### `pwr_model.py`
Forecasts PV system removals due to performance degradation using Weibull CDFs. Outputs forecasts and diagnostic plots by PVCZ and cohort.

### `dmg_model.py`
Models cumulative failures from damage or technical issues. Uses Weibull distribution to simulate failure by year 15.

### `econ_model.py`
Estimates break-even and optimal replacement timing using cash flow and net present value (NPV) modeling. Accounts for degradation, costs, rebates, and electricity rates.

### `scn_model.py`
Combines component models using different scenario weightings. Produces integrated EoL forecasts and plots by scenario.

### `merge.py`
Merges county-level data with GHI and PVCZ assignments based on Karin et al. (2019) spatial overlays.

### `prep_track_sun_params.py`
Processes constants used in modeling: upgrade ratios, system size medians, and average rebates by cohort.

### `electric_rates.py`
Prepares electricity rate forecasts using EIA data and PTC values for PA (from PPL Electric, 2023).

## Data Sources

- **PJM EIS (2025)** – Residential PV system install dataset
- **Karin et al. (2019)** – PV Climate Zones (temperature and humidity maps)
- **DOE/NREL/Statista** – Historical solar cost-per-watt data
- **EIA (2024)** – Residential electricity rates by state and year
- **PPL Electric (2023)** – Price to Compare data for Pennsylvania

## Requirements

- Python 3.7 or higher
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `os`
  - `geopandas`
  - `shapely`

## How to Reproduce

1. **Prepare input data**
   - Place installation data, PVCZ assignments, and supporting datasets into the `data/` directory.
   - Check that filenames match those expected in each script (see docstrings).

2. **Preprocess data** *(optional, depending on workflow)*
   - Run `merge.py` to assign PVCZ and GHI values to counties
   - Run `prep_track_sun_params.py` to print out key cohort-level modeling constants
   - Run `electric_rates.py` to generate custom Pt estimates

3. **Run individual models**
   - Execute `pwr_model.py`, `dmg_model.py`, and `econ_model.py` to generate individual EoL distributions and CSVs

4. **Run scenario integration**
   - Execute `scn_model.py` to generate weighted forecasts and summary plots

5. **Review outputs**
   - Check the `outputs/` directory for CSV summaries and the `outputs/plots/` directory for visuals

Each script prints progress and saves intermediate and final outputs. See in-code comments and docstrings for more detail.

## Credits
- Kathryn Shatto, Athena Kahler, Kayla Tighe
- ENVM 670 Environmental Management Capstone
- University of Maryland Global Campus
- Dr. Sabrina Fu & Sponsor David Comis
