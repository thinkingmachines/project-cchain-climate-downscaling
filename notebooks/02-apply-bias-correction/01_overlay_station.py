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

# Library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
YEAR = 2007

DOMAINS_GEOJSON = RAW_PATH / "domains/downscaling_domains_fixed.geojson"
STATION_LOCATION_CSV = RAW_PATH / "station_data/PAGASA_station_locations.csv"
STATION_DATA_CSV = PROCESSED_PATH / "station_data.csv"
VARS_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.lower()}.nc"
)
VARS = ["precip", "tmax", "tmin"]

# %% [markdown]
# ### Load station location

# %%
station_locations_df = pd.read_csv(STATION_LOCATION_CSV)
station_locations_df.head()
station_lat = station_locations_df.loc[
    station_locations_df["station_name"] == CITY_NAME, "lat"
]
station_lon = station_locations_df.loc[
    station_locations_df["station_name"] == CITY_NAME, "lon"
]

# %% [markdown]
# ### Load station data

# %%
stations_df = pd.read_csv(STATION_DATA_CSV)
station_df = (
    stations_df[stations_df["station"] == CITY_NAME]
    .drop_duplicates()
    .reset_index(drop=True)
    .replace(-999, np.nan)
    .rename(columns={"rainfall": "precip"})
)
station_df.head()

# %%
# bounds_gdf = gpd.read_file(DOMAINS_GEOJSON, driver="GeoJSON")
# station_centroid = (
#     bounds_gdf[bounds_gdf["city"] == CITY_NAME]
#     .to_crs(3857)
#     .centroid
#     .to_crs(4326)
# )
# station_lat = station_centroid.y
# station_lon = station_centroid.x

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
        lon=("lon", station_lon),
        lat=("lat", station_lat),
    ),
    attrs=dict(
        description="Station data",
    ),
)
station_ds

# %%
station_tmin_da = station_ds["tmin"]
station_tmin_da[:, 0, 0].plot(marker="o", linestyle="")

# %%
station_tmax_da = station_ds["tmax"]
station_tmax_da[:, 0, 0].plot(marker="o", linestyle="")

# %%
station_precip_da = station_ds["precip"]
station_precip_da[:, 0, 0].plot(marker="o", linestyle="")

# %% [markdown]
# ### Load gridded data

# %%
gridded_ds = xr.open_dataset(VARS_NC, engine="scipy").sel(band=1)
gridded_ds

# %%
gridded_ds["tmin"].sel(time="2008-07-01", method="nearest").plot()

# %%
station_tmin_da.sel(time="2008-07-01")

# %% [markdown]
# ### Minimum temperature

# %%
gridded_tmin_da = gridded_ds["tmin"].sel(time="2008-07-01", method="nearest")
gridded_tmin_da.plot(vmin=20, vmax=31)
plt.plot(station_lon, station_lat, "o")

# %%
gridded_tmin_station_da = gridded_tmin_da.where(
    (gridded_tmin_da.lat >= (station_lat.item() - 0.125))
    & (gridded_tmin_da.lat <= (station_lat.item() + 0.125))
    & (gridded_tmin_da.lon >= (station_lon.item() - 0.125))
    & (gridded_tmin_da.lon <= (station_lon.item() + 0.125)),
    drop=True,
)
gridded_tmin_station_da.plot(vmin=20, vmax=31)

# %%
gridded_tmin_station_da.plot.hist(bins=15)

# %% [markdown]
# ### Fit distributions

# %%
from scipy.stats import norm

# %%
# x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)

# assume normal distribution with mean centered at station data
# and standard deviation one-tenth of the said value
station_tmin_mean = station_tmin_da.sel(time="2008-07-01").values[0]
station_tmin_std = station_tmin_mean / 10

gridded_tmin_station_arr = gridded_tmin_station_da.values
gridded_tmin_station_mean = np.nanmean(gridded_tmin_station_arr)
gridded_tmin_station_std = np.nanstd(gridded_tmin_station_arr)

# x = np.linspace(station_tmin_mean-4*station_tmin_std,station_tmin_mean+4*station_tmin_std,100)
x_w_nan = gridded_tmin_station_da.values.flatten()
x = x_w_nan[~np.isnan(x_w_nan)]
plt.plot(
    x, norm.pdf(x, loc=station_tmin_mean, scale=station_tmin_std), "o", label="station"
)
plt.plot(
    x,
    norm.pdf(x, loc=gridded_tmin_station_mean, scale=gridded_tmin_station_std),
    "o",
    label="gridded",
)
plt.legend()

# %%
n = gridded_tmin_station_da.values.flatten()
ratio_df = pd.DataFrame(
    dict(
        n=n,
        ratio=(
            norm.pdf(n, loc=station_tmin_mean, scale=station_tmin_std)
            / norm.pdf(n, loc=gridded_tmin_station_mean, scale=gridded_tmin_station_std)
        ),
    )
)

# %%
dist_df = pd.DataFrame(
    dict(
        x=x,
        station_dist=norm.pdf(x, loc=station_tmin_mean, scale=station_tmin_std),
        gridded_dist=norm.pdf(
            x, loc=gridded_tmin_station_mean, scale=gridded_tmin_station_std
        ),
    ),
)
dist_df["dist_ratio"] = dist_df["gridded_dist"] / dist_df["station_dist"]
dist_df


# %%
def custom_replace(da, to_replace, value):
    """
    From https://github.com/pydata/xarray/issues/6377#issue-1173497454
    """
    # Use np.unique to create an inverse index
    flat = da.values.ravel()
    uniques, index = np.unique(flat, return_inverse=True)
    replaceable = np.isin(flat, to_replace)

    # Create a replacement array in which there is a 1:1 relation between
    # uniques and the replacement values, so that we can use the inverse index
    # to select replacement values.
    valid = np.isin(to_replace, uniques, assume_unique=True)
    # Remove to_replace values that are not present in da. If no overlap
    # exists between to_replace and the values in da, just return a copy.
    if not valid.any():
        return da.copy()
    to_replace = to_replace[valid]
    value = value[valid]

    replacement = np.zeros_like(uniques)
    replacement[np.searchsorted(uniques, to_replace)] = value

    out = flat.copy()
    out[replaceable] = replacement[index[replaceable]]
    return da.copy(data=out.reshape(da.shape))


# %%
def correct_gridded(gridded_da: xr.DataArray, station_da: xr.DataArray):
    gridded_arr = gridded_da.values
    gridded_mean = np.nanmean(gridded_arr)
    gridded_std = np.nanstd(gridded_arr)

    station_mean = station_da.values[0]
    station_std = station_mean / 10

    # n_w_nan = gridded_arr.flatten()
    # n = n_w_nan[~np.isnan(n_w_nan)]
    n = np.linspace(station_mean - 4 * station_std, station_mean + 4 * station_std, 100)
    # ratio = (
    #     norm.pdf(n, loc=gridded_mean, scale=gridded_std)
    #     / norm.pdf(n, loc=station_mean, scale=station_std)
    # )
    # ratio_da = custom_replace(gridded_da, n, ratio)
    # print(ratio_da)
    ratio_max = (norm.pdf(n, loc=gridded_mean, scale=gridded_std).max()) / (
        norm.pdf(n, loc=station_mean, scale=station_std).max()
    )
    # print(ratio_max)
    # print(2*np.log(ratio_da*gridded_std/station_std))

    corrected_da = (
        station_std
        * np.sqrt(
            2 * np.log(ratio_max * gridded_std / station_std)
            + ((gridded_da - gridded_mean) / gridded_std) ** 2
        )
        + station_mean
    )
    return corrected_da


corrected_tmin_da = correct_gridded(
    gridded_tmin_station_da, station_da=station_tmin_da.sel(time="2008-07-01")
)

# %%
(corrected_tmin_da - gridded_tmin_station_da).plot()

# %%
gridded_tmin_station_da.plot()
plt.show()

# %%
corrected_tmin_da.plot()

# %%
plt.scatter(
    gridded_tmin_station_da.values.flatten(),
    corrected_tmin_da.values.flatten(),
)
ax = plt.gca()
ax.set_aspect(1)

# %%
gridded_tmin_da[0].values

# %%
gridded_tmin_station_da

# %%
gridded_tmin_station_mean

# %%
(gridded_tmin_station_da - gridded_tmin_station_mean) ** 2

# %% [markdown]
# ### Correct CHIRTS with Li et al. (2019)

# %% [markdown]
# `f_d_n` # gridded
#
# `f_d_n_prime` # station with artificial distribution

# %%
# x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)
x = np.linspace(-100, 100, 100)
plt.plot(x, norm.pdf(x, loc=10, scale=1), "o-")

# %%

# %% [markdown]
# ### Precipitation

# %%
gridded_precip_da = gridded_ds["precip"]
gridded_precip_da.sel(time="2008-07-01", method="nearest")
plt.plot(station_lon, station_lat, "o")

# %%
gridded_precip_da

# %%
gridded_precip_da = gridded_ds["precip"]
gridded_precip_da.isel(time=182).plot()

# %%
gridded_precip_da = gridded_ds["precip"]
gridded_precip_da.sel(time="2008-07-01").plot()
plt.plot(station_lon, station_lat, "o")

# %%
station_precip_da.sel(time="2008-07-01")

# %%
(
    gridded_precip_da.where(
        (gridded_precip_da.lat >= (station_lat.item() - 0.125))
        & (gridded_precip_da.lat <= (station_lat.item() + 0.125))
        & (gridded_precip_da.lon >= (station_lon.item() - 0.125))
        & (gridded_precip_da.lon <= (station_lon.item() + 0.125)),
        drop=True,
    )
    .sel(time="2008-07-01")
    .plot()
)

# %%
(
    gridded_precip_da.where(
        (gridded_precip_da.lat >= (station_lat.item() - 0.125))
        & (gridded_precip_da.lat <= (station_lat.item() + 0.125))
        & (gridded_precip_da.lon >= (station_lon.item() - 0.125))
        & (gridded_precip_da.lon <= (station_lon.item() + 0.125)),
        drop=True,
    )
    .sel(time="2008-07-01")
    .mean()
)

# %% [markdown]
# # Try quantile mapping

# %%
import numpy as np
import xarray as xr
import random
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

# %%
historical_time = xr.cftime_range(
    "1971-01-01", "2000-12-31", freq="D", calendar="noleap"
)
future_time = xr.cftime_range("2001-01-01", "2030-12-31", freq="D", calendar="noleap")

# get_hist_temp_for_lat = lambda val: 273.15 - (val * np.cos(2 * np.pi * historical_time.dayofyear / 365) + 2 * np.random.random_sample((historical_time.size,)) + 273.15 + .1 * (historical_time - historical_time[0]).days / 365)
# get_rand = lambda: np.random.rand() if np.random.rand() > .5 else  -np.random.rand()

# %%
# latitudes = np.arange(23,27,1)
# some_data = [get_hist_temp_for_lat(val) for val in latitudes]
# data = np.array([some_data, np.array(some_data)+1])

# %%
