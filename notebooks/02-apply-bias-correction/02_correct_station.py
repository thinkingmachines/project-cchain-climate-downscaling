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
CITY_NAME = "Dagupan"
DATE = "2008-07-01"
SHOULD_DEBUG = False

PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction"

STATION_NC = CORRECTED_PATH / f"station_{CITY_NAME.lower()}.nc"
GRIDDED_NC = (
    PROCESSED_PATH
    / f"input/chirts_chirps_regridded_interpolated_{CITY_NAME.lower()}.nc"
)
GRIDDED_SUBSET_NC = CORRECTED_PATH / f"gridded_{CITY_NAME.lower()}.nc"

# %% [markdown]
# ### Set run parameters

# %%
variable_params = dict(
    tmin="CHIRTS minimum temperature",
    # tmax="CHIRTS maximum temperature",
    # precip="CHIRPS precipitation",
)

algo_params = [
    dict(
        name="Liu et al. (2019)",
        func=cd.correct_gridded_liu,
    ),
    # dict(
    #    name="Z-Score",
    #    func=cd.correct_gridded_zscore,
    # ),
]

# %% [markdown]
# ### Load data

# %%
station_ds = xr.open_dataset(STATION_NC, engine="scipy")
gridded_ds = xr.open_dataset(GRIDDED_NC, engine="scipy")
gridded_subset_ds = xr.open_dataset(GRIDDED_SUBSET_NC, engine="scipy")

# %%
station_lat = station_ds["lat"].item()
station_lon = station_ds["lon"].item()

# %% [markdown]
# # Apply bias correction

# %%
for var, title in variable_params.items():
    print(f"Now doing {title}")

    gridded_da = gridded_ds[var].sel(time=DATE, method="nearest")
    gridded_subset_da = gridded_subset_ds[var].sel(time=DATE, method="nearest")

    if SHOULD_DEBUG:
        gridded_da.plot()
        plt.plot(station_lon, station_lat, "o")
        plt.show()

        gridded_subset_da.plot()
        plt.plot(station_lon, station_lat, "o")
        plt.show()

        gridded_subset_da.plot.hist(bins=15)
        plt.show()

    for algo_param in algo_params:
        print(f"Now doing {algo_param['name']} bias correction")

        corrected_da = algo_param["func"](
            gridded_subset_da,
            station_da=station_ds[var].sel(time=DATE, method="nearest"),
            std_scale=0.1,
            should_plot=SHOULD_DEBUG,
        )

        if SHOULD_DEBUG:
            plot_min = min([corrected_da.min(), gridded_subset_da.min()]).values
            plot_max = max([corrected_da.max(), gridded_subset_da.max()]).values

            gridded_subset_da.plot(vmin=plot_min, vmax=plot_max)
            plt.title(title)
            plt.show()

            corrected_da.plot(vmin=plot_min, vmax=plot_max)
            plt.title(f"Corrected {title}\n{algo_param['name']}")
            plt.show()

            (corrected_da - gridded_subset_da).plot(cmap="RdBu")
            plt.title(
                f"Difference between corrected and uncorrected\n{title}\n{algo_param['name']}"
            )
            plt.show()

# %%
corrected_da

# %%
