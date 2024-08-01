# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: climate-downscaling
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# Standard imports
from pathlib import Path
import sys

# Library imports
import numpy as np
import xarray as xr

# Util imports
sys.path.append("../../")

# %%
CITY_NAME = "Davao"
RESOLUTION = 0.02
CELL_WIDTH = 3

PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction"

GRIDDED_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.split('_')[0].lower()}.nc"
)
CORRECTED_SUBSET_NC = CORRECTED_PATH / f"corrected_subset_{CITY_NAME.lower()}.nc"
CORRECTED_NC = CORRECTED_PATH / f"corrected_gridded_{CITY_NAME.lower()}.nc"

# %%
gridded_all_ds = xr.open_dataset(GRIDDED_NC).sel(band=1)[["tmin", "tmax", "precip"]]
corrected_subset_all_ds = xr.open_dataset(CORRECTED_SUBSET_NC)

# get common time axis
gridded_aligned_ds, _ = xr.align(
    gridded_all_ds,
    corrected_subset_all_ds,
    join="inner",
)

gridded_ds = gridded_all_ds.sel(time=gridded_aligned_ds["time"])
corrected_subset_ds = corrected_subset_all_ds.sel(time=gridded_aligned_ds["time"])
# gridded_ds["tmin"][0].plot()
# plt.title("CHIRTS Minimum Temperature")
# plt.show()

# %%
gridded_ds.loc[
    dict(
        lat=corrected_subset_ds["lat"],
        lon=corrected_subset_ds["lon"],
    )
] = corrected_subset_ds

# %%
lat_arr = corrected_subset_ds["lat"].to_numpy()
lat_min = lat_arr.min()
lat_max = lat_arr.max()
lat_min_add_arr = np.arange(
    lat_min - (CELL_WIDTH * RESOLUTION), lat_min - RESOLUTION, RESOLUTION
)
lat_max_add_arr = np.arange(
    lat_max + RESOLUTION, lat_max + ((CELL_WIDTH + 1) * RESOLUTION), RESOLUTION
)
lat_extended_arr = np.concatenate(
    (
        lat_min_add_arr,
        lat_arr,
        lat_max_add_arr,
    )
)

lon_arr = corrected_subset_ds["lon"].to_numpy()
lon_min = lon_arr.min()
lon_max = lon_arr.max()
lon_min_add_arr = np.arange(
    lon_min - (CELL_WIDTH * RESOLUTION), lon_min - RESOLUTION, RESOLUTION
)
lon_max_add_arr = np.arange(
    lon_max + RESOLUTION, lon_max + (CELL_WIDTH * RESOLUTION), RESOLUTION
)
lon_extended_arr = np.concatenate(
    (
        lon_min_add_arr,
        lon_arr,
        lon_max_add_arr,
    )
)

# %% [markdown]
# ### Overlay corrected on top of original gridded

# %%
gridded_ds.loc[
    dict(
        lat=corrected_subset_ds["lat"],
        lon=corrected_subset_ds["lon"],
    )
] = corrected_subset_ds
# gridded_ds["tmin"][0].plot()
# plt.title("Corrected on top of CHIRTS Minimum Temperature")
# plt.show()

# %% [markdown]
# ### Smoothen entire domain

# %%
gridded_smoothened_all_ds = gridded_ds.rolling(
    lat=5, lon=5, min_periods=1, center=True
).mean()

# %% [markdown]
# ### Replace center of station influence with corrected

# %%
gridded_smoothened_all_ds.loc[
    dict(
        lat=corrected_subset_ds["lat"][2:-2],
        lon=corrected_subset_ds["lon"][2:-2],
    )
] = corrected_subset_ds.isel(
    lat=range(len(corrected_subset_ds["lat"]))[2:-2],
    lon=range(len(corrected_subset_ds["lon"]))[2:-2],
)
gridded_smoothened_all_ds = gridded_smoothened_all_ds.where(gridded_ds.notnull())

# %% [markdown]
# ### Return the original gridded to the outer areas

# %%
# gridded_smoothened_ds = gridded_ds.copy()
gridded_ds.loc[
    dict(
        lat=gridded_ds["lat"].sel(lat=lat_extended_arr, method="nearest"),
        lon=gridded_ds["lon"].sel(lon=lon_extended_arr, method="nearest"),
    )
] = gridded_smoothened_all_ds.sel(
    lat=lat_extended_arr,
    lon=lon_extended_arr,
    method="nearest",
)
# gridded_ds["tmin"][0].plot()
# plt.title("Smoothened Corrected Minimum Temperature")
# plt.show()

# %%
gridded_ds.to_netcdf(CORRECTED_NC, engine="scipy")

# %%
