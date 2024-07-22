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
import geopandas as gpd
import xarray as xr
import xesmf as xe

# %% [markdown]
# ### Input parameters

# %%
INPUT_PATH = Path("../../data/01-raw")
DEST_PATH = Path("../../data/02-processed")
RESOLUTION = 0.02  # 2 km
CROP_ALLOWANCE_DEG = 13 * RESOLUTION
CITY_NAME = "Dagupan"
YEAR = 2007

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
# ### Align SRTM

# %%
elev_ds = xr.open_dataset(INPUT_PATH / "dem" / f"SRTM_{CITY_NAME}.tiff")
elev_ds = elev_ds.rename({"x": "lon", "y": "lat"})
elev_ds

# %%
elev_regridder = xe.Regridder(elev_ds, ds_grid, "bilinear")
elev_regridder

# %%
elev_regridded_ds = elev_regridder(elev_ds, keep_attrs=True)
elev_regridded_ds = elev_regridded_ds.rename_vars({"band_data": "elevation"})
elev_regridded_ds

# %%
elev_regridded_ds["elevation"].plot()

# %%
del elev_ds

# %%
# elev_regridded_ds = elev_regridded_ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
elev_regridded_ds.to_netcdf(
    DEST_PATH / "input" / f"srtm_regridded_{CITY_NAME.lower()}.nc", engine="scipy"
)

# %% [markdown]
# ## Align NDVI

# %%
from functools import partial


def _preprocess(ds, lon_bnds, lat_bnds):
    return ds.sel(x=slice(*lon_bnds), y=slice(*lat_bnds))


# %%
ndvi_fns = list((INPUT_PATH / "ndvi").glob(f"NDVI_PH_{YEAR}*.nc"))
ndvi_fns

# %%
lon_bnds, lat_bnds = (
    (lon0 - CROP_ALLOWANCE_DEG, lon1 + CROP_ALLOWANCE_DEG),
    (lat1 + CROP_ALLOWANCE_DEG, lat0 - CROP_ALLOWANCE_DEG),
)
partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
ndvi_fns = list((INPUT_PATH / "ndvi").glob(f"NDVI_PH_{YEAR}*.nc"))
ds = xr.open_mfdataset(ndvi_fns, preprocess=partial_func, drop_variables=["EVI"])
ndvi_ds = ds.reindex(y=list(reversed(ds.y)))
ndvi_ds = ndvi_ds.rename({"x": "lon", "y": "lat"})
ndvi_ds = ndvi_ds.rename({"NDVI_gapfill": "ndvi"})
ndvi_ds

# %%
ndvi_regridder = xe.Regridder(ndvi_ds, ds_grid, "bilinear")
ndvi_regridder

# %%
ndvi_regridded_ds = ndvi_regridder(ndvi_ds, keep_attrs=True)
ndvi_regridded_ds

# %%
ndvi_regridded_ds.isel(time=1)["ndvi"].plot()

# %%
ndvi_regridded_ds

# %%
ndvi_regridded_ds.to_netcdf(
    DEST_PATH / "input" / f"ndvi_regridded_{CITY_NAME.lower()}.nc", engine="scipy"
)


# %% [markdown]
# ## Align ERA5

# %%
def _preprocess(ds, lon_bnds, lat_bnds):
    return ds.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))


# %%
era5_fns = list((INPUT_PATH / "era5").glob(f"ERA5_PH_{YEAR}*.nc"))
era5_fns

# %%
lon_bnds, lat_bnds = (
    (lon0 - CROP_ALLOWANCE_DEG, lon1 + CROP_ALLOWANCE_DEG),
    (lat1 + CROP_ALLOWANCE_DEG, lat0 - CROP_ALLOWANCE_DEG),
)
partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
era5_fns = list((INPUT_PATH / "era5").glob(f"ERA5_PH_{YEAR}*.nc"))
ds = xr.open_mfdataset(era5_fns, preprocess=partial_func)
# era5_ds = era_ds.reindex(y=list(reversed(ds.y)))
era5_ds = ds.rename({"longitude": "lon", "latitude": "lat"})
era5_ds

# %%
era5_ds_mean = era5_ds.resample(time="1D").mean()
for variable in [
    v for v in list(era5_ds_mean.variables.keys()) if v not in ["lon", "lat", "time"]
]:
    era5_ds_mean = era5_ds_mean.rename({variable: f"{variable}_mean"})

era5_ds_min = era5_ds.resample(time="1D").min()
for variable in [
    v for v in list(era5_ds_min.variables.keys()) if v not in ["lon", "lat", "time"]
]:
    era5_ds_min = era5_ds_min.rename({variable: f"{variable}_min"})

era5_ds_max = era5_ds.resample(time="1D").max()
for variable in [
    v for v in list(era5_ds_max.variables.keys()) if v not in ["lon", "lat", "time"]
]:
    era5_ds_max = era5_ds_max.rename({variable: f"{variable}_max"})

era_ds_merged = xr.merge([era5_ds_mean, era5_ds_min, era5_ds_max])
era_ds_merged

# %%
era5_regridder = xe.Regridder(era_ds_merged, ds_grid, "bilinear")
era5_regridder

# %%
era5_regridded_ds = era5_regridder(era_ds_merged, keep_attrs=True)
era5_regridded_ds

# %%
era5_regridded_ds.to_netcdf(
    DEST_PATH / "input" / f"era5_regridded_{CITY_NAME.lower()}.nc", engine="scipy"
)


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

# %% [markdown]
# ## Merge all aligned

# %%
all_ds = xr.merge(
    [
        elev_regridded_ds,
        ndvi_regridded_ds,
        era5_regridded_ds,
        chirts_regridded_ds,
        chirps_regridded_ds,
    ]
)
all_ds

# %%
all_ds.isel(time=1)

# %%
all_ds.to_netcdf(
    DEST_PATH / "input" / f"all_vars_regridded_{CITY_NAME.lower()}.nc", engine="scipy"
)
