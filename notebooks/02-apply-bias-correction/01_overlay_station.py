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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Util imports
sys.path.append("../../")
import src.climate_downscaling_utils as cd

# %% [markdown]
# # Overlay station data to grids

# %% [markdown]
# ### Input parameters

# %%
RAW_PATH = Path("../../data/01-raw")
PROCESSED_PATH = Path("../../data/02-processed")
RESULTS_PATH = Path("../../data/03-results")
RESOLUTION = 0.02  # 2 km
CROP_ALLOWANCE_DEG = 13 * RESOLUTION
CITY_NAME = "Dagupan"

DOMAINS_GEOJSON = RAW_PATH / "domains/downscaling_domains_fixed.geojson"
STATION_LOCATION_CSV = RAW_PATH / "station_data/PAGASA_station_locations.csv"
STATION_DATA_CSV = PROCESSED_PATH / "station_data.csv"
VARS_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.lower()}.nc"
)
VARS = ["precip", "tmax", "tmin"]

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
station_tmin_da = station_ds["tmin"]
station_tmax_da = station_ds["tmax"]
station_precip_da = station_ds["precip"]
station_ds

# %% [markdown]
# ## Gridded data

# %% [markdown]
# ### Load gridded data

# %%
gridded_ds = xr.open_dataset(VARS_NC, engine="scipy").sel(band=1)
gridded_ds

# %%
gridded_subset_ds = gridded_ds.where(
    (gridded_ds.lat >= (station_lat - 0.125))
    & (gridded_ds.lat <= (station_lat + 0.125))
    & (gridded_ds.lon >= (station_lon - 0.125))
    & (gridded_ds.lon <= (station_lon + 0.125)),
    drop=True,
)
gridded_subset_ds

# %% [markdown]
# ## Apply correction
# TODO: Split bias correction to a separate notebook

# %%
bias_params = [
    dict(
        name="Liu et al. (2019)",
        func=cd.correct_gridded_liu,
    ),
    dict(
        name="Z-Score",
        func=cd.correct_gridded_zscore,
    ),
]

# %% [markdown]
# ## Minimum temperature

# %% [markdown]
# ### Prepare data

# %%
gridded_tmin_da = gridded_ds["tmin"].sel(time="2008-07-01", method="nearest")
gridded_tmin_da.plot()
plt.plot(station_lon, station_lat, "o")
plt.show()

gridded_subset_tmin_da = gridded_subset_ds["tmin"].sel(
    time="2008-07-01", method="nearest"
)
gridded_subset_tmin_da.plot()
plt.plot(station_lon, station_lat, "o")
plt.show()

gridded_subset_tmin_da.plot.hist(bins=15)
plt.show()

# %% [markdown]
# ### Apply bias correction

# %%
for bias_param in bias_params:
    print(f"Now doing {bias_param['name']} bias correction")

    corrected_tmin_da = bias_param["func"](
        gridded_subset_tmin_da,
        station_da=station_tmin_da.sel(time="2008-07-01"),
        std_scale=0.1,
    )
    plot_min = min([corrected_tmin_da.min(), gridded_subset_tmin_da.min()]).values
    plot_max = max([corrected_tmin_da.max(), gridded_subset_tmin_da.max()]).values

    gridded_subset_tmin_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title("CHIRTS minimum temp")
    plt.show()

    corrected_tmin_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title(f"Corrected CHIRTS minimum temp\n{bias_param['name']}")
    plt.show()

    (corrected_tmin_da - gridded_subset_tmin_da).plot(cmap="RdBu")
    plt.title(
        f"Difference between corrected and uncorrected\nCHIRTS minimum temp\n{bias_param['name']}"
    )
    plt.show()

# %% [markdown]
# ## Maximum temperature

# %% [markdown]
# ### Prepare data

# %%
gridded_tmax_da = gridded_ds["tmax"].sel(time="2008-07-01", method="nearest")
gridded_tmax_da.plot()
plt.plot(station_lon, station_lat, "o")
plt.show()

gridded_subset_tmax_da = gridded_subset_ds["tmax"].sel(
    time="2008-07-01", method="nearest"
)
gridded_subset_tmax_da.plot()
plt.plot(station_lon, station_lat, "o")
plt.show()

gridded_subset_tmax_da.plot.hist(bins=15)
plt.show()

# %% [markdown]
# ### Apply bias correction

# %%
for bias_param in bias_params:
    print(f"Now doing {bias_param['name']} bias correction")

    corrected_tmax_da = bias_param["func"](
        gridded_subset_tmax_da,
        station_da=station_tmax_da.sel(time="2008-07-01"),
        std_scale=0.1,
    )
    plot_min = min([corrected_tmax_da.min(), gridded_subset_tmax_da.min()]).values
    plot_max = max([corrected_tmax_da.max(), gridded_subset_tmax_da.max()]).values

    gridded_subset_tmax_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title("CHIRTS maximum temp")
    plt.show()

    corrected_tmax_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title(f"Corrected CHIRTS maximum temp\n{bias_param['name']}")
    plt.show()

    (corrected_tmax_da - gridded_subset_tmax_da).plot(cmap="RdBu")
    plt.title(
        f"Difference between corrected and uncorrected\nCHIRTS maximum temp\n{bias_param['name']}"
    )
    plt.show()

# %% [markdown]
# ## Precipitation

# %%
gridded_precip_da = gridded_ds["precip"].sel(time="2008-07-01", method="nearest")
gridded_precip_da.plot()
plt.plot(station_lon, station_lat, "o")
plt.show()

gridded_subset_precip_da = gridded_subset_ds["precip"].sel(
    time="2008-07-01", method="nearest"
)
gridded_subset_precip_da.plot()
plt.plot(station_lon, station_lat, "o")
plt.show()

gridded_subset_precip_da.plot.hist(bins=15)
plt.show()

# %% [markdown]
# ### Apply bias correction

# %%
for bias_param in bias_params:
    print(f"Now doing {bias_param['name']} bias correction")

    corrected_precip_da = bias_param["func"](
        gridded_subset_precip_da,
        station_da=station_precip_da.sel(time="2008-07-01"),
        std_scale=0.1,
    )
    plot_min = min([corrected_precip_da.min(), gridded_subset_precip_da.min()]).values
    plot_max = max([corrected_precip_da.max(), gridded_subset_precip_da.max()]).values

    gridded_subset_precip_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title("CHIRPS precipitation")
    plt.show()

    corrected_precip_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title(f"Corrected CHIRPS precipitation\n{bias_param['name']}")
    plt.show()

    (corrected_precip_da - gridded_subset_precip_da).plot(cmap="RdBu")
    plt.title(
        f"Difference between corrected and uncorrected\nCHIRPS precipitation\n{bias_param['name']}"
    )
    plt.show()

# %%
