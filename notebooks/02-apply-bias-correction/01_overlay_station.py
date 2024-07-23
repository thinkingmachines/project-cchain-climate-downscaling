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
from functools import partial

import numpy as np
import geopandas as gpd
import xarray as xr
import xesmf as xe

# %% [markdown]
# # Familiarize with data
#
# 1. Create first plots per city and overlay station data as points
#
# From `notebooks/01-align-grids/02_align_data.ipynb`

# %% [markdown]
# ### Input parameters

# %%
INPUT_PATH = Path("../../data/01-raw")
DEST_PATH = Path("../../data/02-processed")
RESOLUTION = 0.02  # 2 km
CROP_ALLOWANCE_DEG = 13 * RESOLUTION
CITY_NAME = "Dagupan"
YEAR = 2007

(DEST_PATH / "input").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Read bounds

# %%
bounds_gdf = gpd.read_file(
    INPUT_PATH / "domains" / "downscaling_domains_fixed.geojson", driver="GeoJSON"
)
bounds_gdf.head()

# %%
city_bounds_gdf = bounds_gdf[bounds_gdf["city"] == CITY_NAME].copy()
city_bounds_gdf

# %%
lon0, lat0, lon1, lat1 = city_bounds_gdf.total_bounds
lon0, lat0, lon1, lat1

# %% [markdown]
# ### Create grid

# %%
ds_grid = xr.Dataset(
    {
        "lat": (["lat"], np.arange(lat0, lat1 + RESOLUTION, RESOLUTION)),
        "lon": (["lon"], np.arange(lon0, lon1 + RESOLUTION, RESOLUTION)),
    }
)
ds_grid


# %% [markdown]
# ## Align CHIRTS

# %%
def _preprocess(ds, lon_bnds, lat_bnds):
    return ds.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))


# %%
lon_bnds, lat_bnds = (
    (lon0 - CROP_ALLOWANCE_DEG, lon1 + CROP_ALLOWANCE_DEG),
    (lat0 - CROP_ALLOWANCE_DEG, lat1 + CROP_ALLOWANCE_DEG),
)
partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
chirts_fns = list((INPUT_PATH / "chirts").glob(f"CHIRTS_Tmax_PH_{YEAR}*.nc"))
ds = xr.open_mfdataset(chirts_fns, preprocess=partial_func)
chirts_tmax_ds = ds.rename({"longitude": "lon", "latitude": "lat"})
chirts_tmax_ds

# %%
lon_bnds, lat_bnds = (
    (lon0 - CROP_ALLOWANCE_DEG, lon1 + CROP_ALLOWANCE_DEG),
    (lat0 - CROP_ALLOWANCE_DEG, lat1 + CROP_ALLOWANCE_DEG),
)
partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
chirts_fns = list((INPUT_PATH / "chirts").glob(f"CHIRTS_Tmin_PH_{YEAR}*.nc"))
ds = xr.open_mfdataset(chirts_fns, preprocess=partial_func)
chirts_tmin_ds = ds.rename({"longitude": "lon", "latitude": "lat"})
chirts_tmin_ds

# %%
chirts_regridder = xe.Regridder(chirts_tmin_ds, ds_grid, "bilinear")
chirts_regridder

# %%
chirts_tmax_regridded_ds = chirts_regridder(chirts_tmax_ds, keep_attrs=True)
chirts_tmax_regridded_ds

# %%
chirts_tmin_regridded_ds = chirts_regridder(chirts_tmin_ds, keep_attrs=True)
chirts_tmin_regridded_ds

# %%
chirts_regridded_ds = xr.merge([chirts_tmin_regridded_ds, chirts_tmax_regridded_ds])
for variable in [
    v
    for v in list(chirts_regridded_ds.variables.keys())
    if v not in ["lon", "lat", "time"]
]:
    chirts_regridded_ds = chirts_regridded_ds.rename({variable: variable.lower()})
chirts_regridded_ds

# %%
chirts_tmin_ds.isel(time=100)["Tmin"].plot()

# %%
chirts_regridded_ds.isel(time=100)["tmin"].plot()

# %%
chirts_regridded_ds.to_netcdf(
    DEST_PATH / "input" / f"chirts_regridded_{CITY_NAME.lower()}.nc", engine="scipy"
)

# %% [markdown]
# ## Align CHIRPS

# %%
lon_bnds, lat_bnds = (
    (lon0 - CROP_ALLOWANCE_DEG, lon1 + CROP_ALLOWANCE_DEG),
    (lat0 - CROP_ALLOWANCE_DEG, lat1 + CROP_ALLOWANCE_DEG),
)
partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
chirps_fns = list((INPUT_PATH / "chirps").glob(f"CHIRPS_PH_{YEAR}*.nc"))
ds = xr.open_mfdataset(chirps_fns, preprocess=partial_func)
chirps_ds = ds.rename({"longitude": "lon", "latitude": "lat"})
chirps_ds

# %%
chirps_regridder = xe.Regridder(chirps_ds, ds_grid, "bilinear")
chirps_regridder

# %%
chirps_regridded_ds = chirps_regridder(chirps_ds, keep_attrs=True)
chirps_regridded_ds

# %%
chirps_ds.isel(time=100)["precip"].plot()

# %%
chirps_regridded_ds.isel(time=100)["precip"].plot()

# %%
chirps_regridded_ds.to_netcdf(
    DEST_PATH / "input" / f"chirps_regridded_{CITY_NAME.lower()}.nc", engine="scipy"
)

# %%
