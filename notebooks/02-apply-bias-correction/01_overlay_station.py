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
import pandas as pd
import xarray as xr

# Util imports
sys.path.append("../../")

# %% [markdown]
# # Overlay station data to grids

# %% [markdown]
# ### Input parameters

# %%
CITY_NAME = "Dagupan"
VARS = ["precip", "tmax", "tmin"]
STATION_RESOLUTION_DEGREES = 0.25

RAW_PATH = Path("../../data/01-raw")
PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction"
CORRECTED_PATH.mkdir(parents=True, exist_ok=True)

DOMAINS_GEOJSON = RAW_PATH / "domains/downscaling_domains_fixed.geojson"
STATION_LOCATION_CSV = RAW_PATH / "station_data/PAGASA_station_locations.csv"
STATION_DATA_CSV = PROCESSED_PATH / "station_data.csv"

STATION_NC = CORRECTED_PATH / f"station_{CITY_NAME.lower()}.nc"
GRIDDED_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.lower()}.nc"
)
GRIDDED_SUBSET_NC = CORRECTED_PATH / f"gridded_{CITY_NAME.lower()}.nc"

# %% [markdown]
# ## Station data

# %% [markdown]
# ### Load station location

# %%
station_locations_df = pd.read_csv(STATION_LOCATION_CSV)
station_locations_df.head()
station_lats = station_locations_df.loc[
    station_locations_df["station_name"] == CITY_NAME, "lat"
]
station_lons = station_locations_df.loc[
    station_locations_df["station_name"] == CITY_NAME, "lon"
]
station_lat = station_lats.item()
station_lon = station_lons.item()

# %% [markdown]
# ### Load station data

# %%
stations_df = pd.read_csv(STATION_DATA_CSV)
station_df = (
    stations_df[stations_df["station"] == CITY_NAME]
    .drop_duplicates()
    .replace(-999, np.nan)
    .rename(columns={"rainfall": "precip"})
    .sort_values("date")
    .reset_index(drop=True)
)
station_df.head()

# %% [markdown]
# ### Arrange as a Dataset

# %%
station_ds = xr.Dataset(
    data_vars={
        var: (
            ["time", "lat", "lon"],
            station_df[var].to_numpy().reshape((len(station_df["date"]), 1, 1)),
        )
        for var in VARS
    },
    coords=dict(
        time=("time", pd.DatetimeIndex(station_df["date"])),
        lon=("lon", station_lons),
        lat=("lat", station_lats),
    ),
    attrs=dict(
        description="Station data",
    ),
)
station_ds

# %%
station_ds.to_netcdf(STATION_NC, engine="scipy")

# %% [markdown]
# ## Gridded data

# %% [markdown]
# ### Load gridded data

# %%
gridded_ds = xr.open_dataset(GRIDDED_NC, engine="scipy").sel(band=1)
gridded_ds

# %%
station_buffer = STATION_RESOLUTION_DEGREES / 2
gridded_subset_ds = gridded_ds.where(
    (gridded_ds.lat >= (station_lat - station_buffer))
    & (gridded_ds.lat <= (station_lat + station_buffer))
    & (gridded_ds.lon >= (station_lon - station_buffer))
    & (gridded_ds.lon <= (station_lon + station_buffer)),
    drop=True,
)
gridded_subset_ds

# %%
gridded_subset_ds.to_netcdf(GRIDDED_SUBSET_NC, engine="scipy")

# %%
