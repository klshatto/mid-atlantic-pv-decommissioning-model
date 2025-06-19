"""
Mid-Atlantic County Borders Merge with PVCZ Data Set (merge.py)

Cleans, merges, and spatially joins Mid-Atlantic county data with Karin GHI point data to assign
county-level mean GHI values and validate PVCZ labeling. Missing GHI values are filled by spatial
proximity (nearest neighbor) or by zone-level averages.

Steps:
- Cleans county shapefile and county list from midatlantic_county_pvcz.csv
- Joins Karin GHI point data to counties using spatial overlay
- Aggregates GHI_mean per county, fills missing values hierarchically
- Produces maps of GHI and PVCZ for the Mid-Atlantic region
- Outputs merged table for modeling and zone-average GHI summary

Expected Inputs:
- ../data/pvcz_data.csv  
    → Karin et al. (2019) GHI and PVCZ point dataset
- ../data/shapefiles/cb_2023_us_county_500k.shp  
    → U.S. county shapefile
- ../data/midatlantic_county_pvcz.csv  
    → List of Mid-Atlantic counties with assigned PVCZ

Outputs:
- ../outputs/midatlantic_county_with_ghi.csv  
    → Final county-level table with PVCZ and GHI values
- ../outputs/pvcz_ghi_means.csv  
    → Zone-level average GHI for PVCZs
- ../outputs/plots/mid_atlantic_GHI.png  
    → Map of GHI values by county
- ../outputs/plots/mid_atlantic_PVCZ.png  
    → Map of PVCZs by county

Authors: Kathryn Shatto (script author), Athena Kahler, Kayla Tighe
ENVM 670 Environmental Management Capstone  
University of Maryland Global Campus  
Dr. Sabrina Fu & Sponsor David Comis  
Spring 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Load Karin GHI points ---
karin_df = pd.read_csv('../data/pvcz_data.csv')  # Karin data set
karin_gdf = gpd.GeoDataFrame(
    karin_df,
    geometry=gpd.points_from_xy(karin_df.lon, karin_df.lat),
    crs="EPSG:4326"
)

# --- Load county shapefile ---
county_gdf = gpd.read_file('../data/shapefiles/cb_2023_us_county_500k.shp')
county_gdf['County'] = county_gdf['NAME'].str.replace(' County', '', regex=False).str.replace(' City', '', regex=False)
county_gdf['State'] = county_gdf['STUSPS']

# --- Get state border layer ---
midatlantic_states = ['DC', 'DE', 'MD', 'NC', 'NJ', 'PA', 'VA', 'WV']
state_boundaries = county_gdf[county_gdf['STUSPS'].isin(midatlantic_states)].dissolve(by='State')

# --- Load and clean your Mid-Atlantic counties list ---
midatlantic_df = pd.read_csv('../data/midatlantic_county_pvcz.csv')
midatlantic_df['County'] = midatlantic_df['County'].str.replace(' County', '', regex=False).str.replace(' City', '', regex=False)

def clean_county(name):
    if pd.isnull(name): return ""
    name = name.lower()
    name = name.replace("'", "")    # remove apostrophes
    name = name.replace(".", "")    # remove periods
    name = name.replace(" ", "")    # remove spaces
    return name.strip()

county_gdf['County_clean'] = county_gdf['County'].apply(clean_county)
midatlantic_df['County_clean'] = midatlantic_df['County'].apply(clean_county)

# --- Merge to get only Mid-Atlantic counties/cities with geometry ---
merged = county_gdf.merge(midatlantic_df, on=['State', 'County_clean'], how='inner')
merged['PVCZ'] = merged['PVCZ']
merged['County'] = merged['County_y']
print("After merge:", merged.shape)

# --- Spatial join: assign each Karin point to a county ---
# Project to same CRS if needed
karin_gdf = karin_gdf.to_crs(county_gdf.crs)
joined = gpd.sjoin(karin_gdf, county_gdf, how="inner", predicate='within')

# --- Compute county-level GHI_mean ---
county_ghi = joined.groupby(['State', 'County']).agg({'GHI_mean': 'mean'}).reset_index()

# --- Merge county_ghi to your modeling table ---
midatlantic_with_ghi = pd.merge(merged, county_ghi, on=['State', 'County'], how='left')
midatlantic_with_ghi = gpd.GeoDataFrame(midatlantic_with_ghi, geometry='geometry', crs=county_gdf.crs)
print(midatlantic_with_ghi['GHI_mean'].isna().sum(), "counties/cities missing GHI values")

# Get counties with missing GHI
missing = midatlantic_with_ghi[midatlantic_with_ghi['GHI_mean'].isna()]
not_missing = midatlantic_with_ghi.dropna(subset=['GHI_mean'])

pvcz_ghi = midatlantic_with_ghi.groupby('PVCZ')['GHI_mean'].mean().reset_index()

# Fill missing county/city GHI with nearest neighbor, then (if needed) with PVCZ mean as fallback.
# Use UTM 18N (EPSG:26918) for the Mid-Atlantic region
projected_crs = "EPSG:26918"
midatlantic_with_ghi = midatlantic_with_ghi.to_crs(projected_crs)
not_missing = not_missing.to_crs(projected_crs)
missing = missing.to_crs(projected_crs)
for idx, row in missing.iterrows():
    # Find nearest county with a GHI value
    distances = not_missing.geometry.distance(row.geometry)
    nearest_idx = distances.idxmin()
    midatlantic_with_ghi.at[idx, 'GHI_mean'] = not_missing.loc[nearest_idx, 'GHI_mean']
if midatlantic_with_ghi['GHI_mean'].isna().sum() > 0:
    # Fill any remaining with PVCZ means
    midatlantic_with_ghi = pd.merge(
        midatlantic_with_ghi, pvcz_ghi, on='PVCZ', how='left', suffixes=('', '_zone')
    )
    midatlantic_with_ghi['GHI_mean'] = midatlantic_with_ghi['GHI_mean'].fillna(midatlantic_with_ghi['GHI_mean_zone'])
    midatlantic_with_ghi = midatlantic_with_ghi.drop(columns=['GHI_mean_zone'])

assert midatlantic_with_ghi['GHI_mean'].isna().sum() == 0, "Some counties/cities still missing GHI!"

# After all your merges and assignments:
final_output = pd.merge(midatlantic_df, midatlantic_with_ghi[['State', 'County', 'PVCZ', 'GHI_mean']], 
                        on=['State', 'County'], how='left')
# Now check for remaining NAs
print(f"After all assignments, still missing: {final_output['GHI_mean'].isna().sum()} counties/cities")

# --- Optionally, assign PVCZ-level averages ---
pvcz_ghi = midatlantic_with_ghi.groupby('PVCZ')['GHI_mean'].mean().reset_index()

# --- Export result for your modeling code ---
midatlantic_with_ghi.to_csv('../outputs/midatlantic_county_with_ghi.csv', index=False)
pvcz_ghi.to_csv('../outputs/pvcz_ghi_means.csv', index=False)

# Plot GHI map
plot_gdf_ghi = county_gdf.merge(
    midatlantic_with_ghi[['State', 'County', 'GHI_mean']],
    on=['State', 'County'],
    how='right'
)
fig, ax = plt.subplots(figsize=(10, 8))
plot_gdf_ghi.plot(
    column='GHI_mean',
    ax=ax,
    legend=True,
    cmap='viridis',
    legend_kwds={'label': "GHI (kW/m²/day)"}
)
ax.set_title("County-level Mean GHI in the Mid-Atlantic", fontsize=14)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
plt.savefig("../outputs/plots/mid_atlantic_GHI.png")
plt.show()

# Plot PVCZ map
plot_gdf_pvcz = county_gdf.merge(
    midatlantic_with_ghi[['State', 'County', 'PVCZ']],
    on=['State', 'County'],
    how='right'
)
PVCZ_COLORS = {
    "T6:H4": "#93ff93",   # light green
    "T7:H4": "#ffff91",   # yellow
    "T8:H4": "#ffbb6c",   # orange
}
plot_gdf_pvcz['color'] = plot_gdf_pvcz['PVCZ'].map(PVCZ_COLORS)
plot_gdf_nonan = plot_gdf_pvcz[plot_gdf_pvcz['color'].notna()]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_gdf_nonan.plot(ax=ax, color=plot_gdf_nonan['color'], linewidth=0.5, edgecolor='gray')
state_boundaries.boundary.plot(ax=ax, color='black', linewidth=2.5)
legend_patches = [
    mpatches.Patch(color=color, label=pvcz)
    for pvcz, color in PVCZ_COLORS.items()
]
ax.legend(handles=legend_patches, title="PVCZ", title_fontsize=20, loc='upper left', fontsize=18)
ax.set_axis_off()
plt.title("Photovoltaic Climate Zones for the Mid-Atlantic Region", fontsize=22)
plt.tight_layout()
plt.savefig("../outputs/plots/mid_atlantic_PVCZ.png", dpi=300)
plt.show()

print("Unique PVCZs:", plot_gdf_pvcz['PVCZ'].unique())
print("Rows missing color:", plot_gdf_pvcz['color'].isna().sum())
