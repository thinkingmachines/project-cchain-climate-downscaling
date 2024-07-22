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
from pathlib import Path

import numpy as np
import xarray as xr

from loguru import logger
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %%
INPUT_PATH = Path("../../data/02-processed")
RESOLUTION = 0.02  # 2 km
CROP_ALLOWANCE_DEG = 13 * RESOLUTION
CITY_NAME = "Dagupan"
YEAR = 2007

# %% [markdown]
# ## Read aligned file

# %%
input_ds = xr.open_dataset(
    INPUT_PATH / "input" / f"all_vars_regridded_{CITY_NAME.lower()}.nc"
)
input_ds

# %% [markdown]
# ### Interpolate across lon and lat

# %%
input_regridded_ds = input_ds.copy()

# %%
mask = xr.where(input_ds["elevation"] > 0, 1, 0)
for variable in [
    v
    for v in list(input_ds.variables.keys())
    if v not in ["lon", "lat", "time", "spatial_ref", "band"]
]:
    input_regridded_ds[variable] = (
        input_regridded_ds[variable]
        .interpolate_na(dim="lon", method="slinear")
        .interpolate_na(dim="lat", method="slinear")
    )
    # Apply mask
    input_regridded_ds[variable] = input_regridded_ds[variable] / mask
    # Inf as nan
    input_regridded_ds = input_regridded_ds.where(
        np.isfinite(input_regridded_ds), np.nan
    )
input_regridded_ds

# %% [markdown]
# ## Save interpolated file

# %%
input_regridded_ds.to_netcdf(
    INPUT_PATH / "input" / f"all_vars_regridded_interpolated_{CITY_NAME.lower()}.nc",
    engine="scipy",
)

# %% [markdown]
# ## Plot data to check interpolation

# %%
# raw
ds_list = []
INPUT_RAW_DIR = Path("../../data/01-raw/")
era_ds = xr.open_dataset(INPUT_RAW_DIR / "era5" / f"ERA5_PH_{YEAR}04.nc")
chirts_tmin_ds = xr.open_dataset(INPUT_RAW_DIR / "chirts" / f"CHIRTS_Tmin_PH_{YEAR}.nc")
chirts_tmax_ds = xr.open_dataset(INPUT_RAW_DIR / "chirts" / f"CHIRTS_Tmax_PH_{YEAR}.nc")
chirps_ds = xr.open_dataset(INPUT_RAW_DIR / "chirps" / f"CHIRPS_PH_{YEAR}.nc")
ndvi_ds = xr.open_dataset(INPUT_RAW_DIR / "ndvi" / f"NDVI_PH_{YEAR}04.nc")
strm_ds = xr.open_dataset(INPUT_RAW_DIR / "dem" / f"SRTM_{CITY_NAME}.tiff")
ds_list = [era_ds, chirts_tmin_ds, chirts_tmax_ds, chirps_ds, ndvi_ds, strm_ds]

# %%
time_idx = 100
nrow = 4
ncol = 7

variables = [
    v
    for v in list(input_ds.variables.keys())
    if v not in ["lon", "lat", "time", "banc", "spatial_ref"]
]
fig, axes = plt.subplots(
    nrow, ncol, figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()}
)
for i, var_name in enumerate(variables):
    row = i // ncol
    col = i % ncol
    # Plot variable on corresponding subplot
    try:
        # data = input_ds.isel(time=time_idx)
        data = input_ds.isel(time=time_idx)
        # data = data.interpolate_na(dim="lon", method='slinear').interpolate_na(dim="lat", method='slinear')
        # mask = xr.where(data['elevation']>0,1,0)
        data = data[var_name]  # /mask
        data.plot(
            ax=axes[row, col],
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            add_colorbar=False,
            add_labels=False,
            xticks=None,
            yticks=None,
        )
        axes[row, col].set_title(var_name)
        axes[row, col].coastlines()
    except Exception as e:
        logger.error(f"Exception raised: {e}")
        continue

for i in range(len(variables) - 1, nrow * ncol):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()

# %%
time_idx = 100
nrow = 4
ncol = 7

variables = [
    v
    for v in list(input_ds.variables.keys())
    if v not in ["lon", "lat", "time", "banc", "spatial_ref"]
]
fig, axes = plt.subplots(
    nrow, ncol, figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()}
)
for i, var_name in enumerate(variables):
    row = i // ncol
    col = i % ncol
    # Plot variable on corresponding subplot
    try:
        # data = input_ds.isel(time=time_idx)
        data = input_regridded_ds.isel(time=time_idx)
        # data = data.interpolate_na(dim="lon", method='slinear').interpolate_na(dim="lat", method='slinear')
        mask = xr.where(data["elevation"] > 0, 1, 0)
        data = data[var_name] / mask
        data.plot(
            ax=axes[row, col],
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            add_colorbar=False,
            add_labels=False,
            xticks=None,
            yticks=None,
        )
        axes[row, col].set_title(var_name)
        axes[row, col].coastlines()
    except Exception as e:
        logger.error(f"Exception raised: {e}")
        continue

for i in range(len(variables) - 1, nrow * ncol):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()

# %%
