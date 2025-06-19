"""
Damage and Technical Failures Model (dmg_model.py)

Forecasts cumulative failure probability for residential PV systems in the Mid-Atlantic 
region due to damage and technical failures, using a Weibull distribution fitted to 
industry data and scaled to match observed early-loss rates.

- Models cumulative probability of EoL due to non-performance (damage/failure)
- Uses Weibull parameters (α, β) from literature/industry estimates
- Outputs annual cumulative failure probabilities and a diagnostic plot

Expected input: None (parameters are hardcoded; standalone script)

Outputs:
- ../outputs/dmg_model_cumulative_failure.csv
    → Table of yearly cumulative failure probabilities (1–15 years)
- ../outputs/plots/dmg_weibull.png
    → Weibull CDF plot (cumulative failure % over time)

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe

ENVM 670 Environmental Management Capstone
University of Maryland Global Campus
Dr. Sabrina Fu & Sponsor David Comis
Spring 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import os
os.makedirs("../outputs/plots", exist_ok=True)

# Set global plot style
import seaborn as sns
sns.set(style="whitegrid", rc={"grid.color": "lightgray", "grid.linestyle": "--"})

# === 1. DEFINE WEIBULL PARAMETERS (FROM LITERATURE) ===

alpha = 2.4928      # Shape parameter (matches "early loss" scenario from Tan et al. and IRENA/IEA-PVPS)
beta = 6.9085       # Scale parameter (scaled so CDF reaches 100% at 15 years per Tan et al., Table 6)

# === 2. CALCULATE CUMULATIVE FAILURE PROBABILITIES ===

years = np.arange(1, 16)  # 1 to 15 years
cdf = weibull_min.cdf(years, c=alpha, scale=beta)  # Cumulative probability at each year

# Create DataFrame for results
weibull_df = pd.DataFrame({
    "Year": years,
    "Cumulative Failure Probability (%)": np.round(cdf * 100, 2)
})

# Save to CSV
weibull_df.to_csv("../outputs/dmg_model_cumulative_failure.csv", index=False)
print("Saved cumulative failure table to dmg_model_cumulative_failure.csv")

# === 3. PLOT WEIBULL CDF ===

plt.figure(figsize=(8, 5))
plt.plot(years, cdf * 100, marker='o')
plt.title('Cumulative Failure Probability – Damage & Technical Failures Model')
plt.xlabel('Years Since Installation')
plt.ylabel('Cumulative Failure Probability (%)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("../outputs/plots/dmg_weibull.png")
plt.show()
