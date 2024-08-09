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
import pandas as pd
import scipy.stats as sc
from sklearn.metrics import mean_squared_error
import xarray as xr

# Util imports
sys.path.append("../../")
import src.climate_downscaling_utils as cd

# %% [markdown]
# # Correct station data
#
# This notebook applies bias correction algorithms on gridded data using station data.
#
# **Prerequisite**: Run `notebooks/02-apply-bias-correction/01_overlay_station.ipynb`

# %% [markdown]
# ### Set input parameters

# %%
CITY_NAME = "Zamboanga"  # with suffix for Cagayan De Oro

DATE = "2008-07-01"  # sample date for debugging
YEARS = [2007, 2008, 2009, 2016, 2017, 2018]
SHOULD_DEBUG = False
PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction-radial-optimized"
CORRECTED_PATH.mkdir(parents=True, exist_ok=True)

STATION_NC = CORRECTED_PATH / f"station_{CITY_NAME.lower()}.nc"
GRIDDED_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.split('_')[0].lower()}.nc"
)
GRIDDED_SUBSET_NC = CORRECTED_PATH / f"gridded_{CITY_NAME.lower()}.nc"

# %% [markdown]
# ### Set run parameters
# - `plot_offset`: margin for x and y bounds

# %%
variable_params = dict(
    tmin="CHIRTS minimum temperature",
    tmax="CHIRTS maximum temperature",
    precip="CHIRPS precipitation",
)

method_params = dict(
    tmin="linear_scaling",
    tmax="linear_scaling",
    precip="linear_scaling",
)

algo_params = dict(
    linear_scaling=dict(
        name="Linear Scaling",
        func=cd.correct_gridded_cmethods,
        group="time.month",
    ),
    variance_scaling=dict(
        name="Variance Scaling",
        func=cd.correct_gridded_cmethods,
        group="time.month",
    ),
    quantile_mapping=dict(
        name="Quantile Mapping",
        func=cd.correct_gridded_cmethods,
        n_quantiles=10,
    ),
    detrended_quantile_mapping=dict(
        name="Detrended Quantile Mapping",
        func=cd.correct_gridded_cmethods,
        n_quantiles=1_000,
    ),
)

# refer to defaults
for method, algo_param in algo_params.items():
    if "group" not in algo_param.keys():
        algo_param["group"] = "time.month"
    if "n_quantiles" not in algo_param.keys():
        algo_param["n_quantiles"] = 1_000


scatterplot_params = dict(
    tmin=dict(
        variable_name="Minimum Temperature",
        units="°C",
        plot_offset=2.5,
        color="firebrick",
    ),
    tmax=dict(
        variable_name="Maximum Temperature",
        units="°C",
        plot_offset=1,
        color="firebrick",
    ),
    precip=dict(
        variable_name="Precipitation",
        units="mm/day",
        plot_offset=10,
        color="dodgerblue",
    ),
)

# %% [markdown]
# ### Load data

# %%
station_ds = xr.open_dataset(STATION_NC, engine="scipy")
gridded_ds = xr.open_dataset(GRIDDED_NC, engine="scipy")
gridded_subset_ds = xr.open_dataset(GRIDDED_SUBSET_NC, engine="scipy")

# %%
gridded_subset_ds

# %%
station_lat = station_ds["lat"].item()
station_lon = station_ds["lon"].item()

# %% [markdown]
# # Apply bias correction

# %%
stats_list = []
corrected_ds = xr.Dataset(data_vars=None)
for var, title in variable_params.items():
    print(f"Now doing {title}")

    if var == "precip":
        gridded_da = gridded_ds.isel(time=gridded_ds.time.dt.year.isin(YEARS))[var]
        gridded_subset_da = gridded_subset_ds.isel(
            time=gridded_subset_ds.time.dt.year.isin(YEARS)
        )[var]
        station_da = station_ds.isel(time=station_ds.time.dt.year.isin(YEARS))[var]
    else:
        gridded_da = gridded_ds.isel(time=gridded_ds.time.dt.year.isin(YEARS))[var]  #
        gridded_subset_da = gridded_subset_ds.isel(
            time=gridded_subset_ds.time.dt.year.isin(YEARS)
        )[var]
        station_da = station_ds.isel(time=station_ds.time.dt.year.isin(YEARS))[var]

    if SHOULD_DEBUG:
        gridded_da.sel(time=DATE, method="nearest").plot()
        plt.plot(station_lon, station_lat, "o")
        plt.show()

        gridded_subset_da.sel(time=DATE, method="nearest").plot()
        plt.plot(station_lon, station_lat, "o")
        plt.show()

        gridded_subset_da.plot.hist(bins=15)
        plt.show()

    algo_param = algo_params[method_params[var]]
    print(f"Now doing {algo_param['name']} bias correction")

    if algo_param["name"] == "Liu et al. (2019)" or algo_param["name"] == "Z-Score":
        corrected_da = algo_param["func"](
            gridded_subset_da.sel(time=DATE, method="nearest"),
            station_da=station_da.sel(time=DATE, method="nearest"),
            std_scale=0.1,
            should_plot=SHOULD_DEBUG,
        )
    else:
        corrected_da = algo_param["func"](
            gridded_da=gridded_subset_da,
            station_da=station_da[:, 0, 0].drop_vars(["lat", "lon"]),
            method=method_params[var],
            n_quantiles=algo_param["n_quantiles"],
            group=algo_param["group"],
            should_plot=SHOULD_DEBUG,
        )

    corrected_ds[var] = corrected_da

    station_aligned_da, corrected_aligned_da = xr.align(
        station_da.mean(dim=["lat", "lon"], skipna=True).dropna(dim="time"),
        corrected_da.mean(dim=["lat", "lon"], skipna=True).dropna(dim="time"),
        join="inner",
    )

    corr, pval = sc.pearsonr(station_aligned_da, corrected_aligned_da)
    rmse = mean_squared_error(station_aligned_da, corrected_aligned_da, squared=False)

    stats_list.append(
        dict(var=var, method=method_params[var], corr=corr, pval=pval, rmse=rmse)
    )

    if SHOULD_DEBUG:
        gridded_subset_slice_da = gridded_subset_da.sel(time=DATE, method="nearest")
        corrected_slice_da = corrected_da.sel(time=DATE, method="nearest")

        plot_min = min([corrected_slice_da.min(), gridded_subset_slice_da.min()]).values
        plot_max = max([corrected_slice_da.max(), gridded_subset_slice_da.max()]).values

        gridded_subset_slice_da.plot(vmin=plot_min, vmax=plot_max)
        plt.title(title)
        plt.show()

        corrected_slice_da.plot(vmin=plot_min, vmax=plot_max)
        plt.title(f"Corrected {title}\n{algo_param['name']}")
        plt.show()

        abs(corrected_slice_da - gridded_subset_slice_da).plot(cmap="RdYlGn")
        plt.title(
            f"Difference between corrected and uncorrected\n{title}\n{algo_param['name']}"
        )
        plt.show()
stats_df = pd.DataFrame(stats_list)

# %%
corrected_ds.to_netcdf(
    CORRECTED_PATH / f"corrected_subset_{CITY_NAME.lower()}.nc",
    engine="scipy",
)

# %%
stats_df.to_parquet(CORRECTED_PATH / f"stats_{CITY_NAME.lower()}.parquet")
stats_df
