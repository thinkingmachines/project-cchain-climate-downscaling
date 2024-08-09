# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Util imports
sys.path.append("../../")

# %% [markdown]
# # Overlay station data to grids
# Prepare gridded data to match station area of influence

# %% [markdown]
# ### Input parameters

# %%
CITY_NAME = "Davao"
STATION_NAMES = ["Davao City"]
SUFFIX = ""  # add underscore, only for Cagayan de Oro
VARS = ["precip", "tmax", "tmin"]
STATION_RESOLUTION_DEGREES = 0.25

RAW_PATH = Path("../../data/01-raw")
PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction-radial-optimized"
CORRECTED_PATH.mkdir(parents=True, exist_ok=True)

DOMAINS_GEOJSON = RAW_PATH / "domains/downscaling_domains_fixed.geojson"
STATION_LOCATION_CSV = RAW_PATH / "station_data/PAGASA_station_locations.csv"
STATION_DATA_CSV = PROCESSED_PATH / "station_data.csv"

STATION_NC = CORRECTED_PATH / f"station_{CITY_NAME.lower()}{SUFFIX.lower()}.nc"
GRIDDED_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.lower()}.nc"
)
GRIDDED_SUBSET_NC = CORRECTED_PATH / f"gridded_{CITY_NAME.lower()}{SUFFIX.lower()}.nc"

# %% [markdown]
# ## Station data

# %% [markdown]
# ### Load station location

# %%
station_locations_df = pd.read_csv(STATION_LOCATION_CSV)
station_locations_df.head()
station_lats = station_locations_df.loc[
    station_locations_df["station_name"].isin(STATION_NAMES), "lat"
]
station_lons = station_locations_df.loc[
    station_locations_df["station_name"].isin(STATION_NAMES), "lon"
]
station_lat = station_lats.item() if len(station_lats) == 1 else station_lats.mean()
station_lon = station_lons.item() if len(station_lons) == 1 else station_lons.mean()

# %%
station_locations_df

# %% [markdown]
# ### Load station data

# %%
stations_df = pd.read_csv(STATION_DATA_CSV)
station_df = (
    stations_df[stations_df["station"].isin(STATION_NAMES)]
    .drop_duplicates()
    .replace(-999, np.nan)
    .rename(columns={"rainfall": "precip"})
    .groupby(["station", "date"])
    .mean()
    .sort_values("date")
    .reset_index()
)
station_df.head()

# %% [markdown]
# ### Arrange as a Dataset

# %%
station_ds = (
    xr.Dataset(
        data_vars={
            var: (
                ["time", "lat"],
                station_df[var]
                .to_numpy()
                .reshape((len(station_df["date"].unique()), len(station_lats))),
            )
            for var in VARS
        },
        coords=dict(
            time=("time", pd.DatetimeIndex(station_df["date"].unique())),
            lat=("lat", station_lats),
        ),
        attrs=dict(
            description="Station data",
        ),
    )
    .expand_dims(lon=len(station_lons))
    .assign_coords(lon=station_lons)
    .transpose("time", "lat", "lon")
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
gridded_ds

# %%
station_lat, station_lon

# %%
station_buffer = STATION_RESOLUTION_DEGREES / 2
gridded_subset_ds = gridded_ds.where(
    (
        (gridded_ds.lat - station_lat) ** 2 + (gridded_ds.lon - station_lon) ** 2
        <= station_buffer**2
    ),
    drop=True,
)
gridded_subset_ds

# %%
gridded_subset_ds["tmin"].sel(time="2007-01-01").plot()
ax = plt.gca()
ax.set_aspect(1)
plt.show()

# %%
GRIDDED_SUBSET_NC

# %%
gridded_subset_ds.to_netcdf(GRIDDED_SUBSET_NC, engine="scipy")

# %%
