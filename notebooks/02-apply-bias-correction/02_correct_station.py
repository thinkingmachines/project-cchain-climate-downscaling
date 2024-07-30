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
CITY_NAME = "Dagupan"
DATE = "2008-07-01"
SHOULD_DEBUG = True
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
        name="Quantile Delta Mapping",
        func=cd.correct_gridded_quantile_mapping,
    ),
    # dict(
    #     name="Liu et al. (2019)",
    #     func=cd.correct_gridded_liu,
    # ),
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

    gridded_da = gridded_ds[var]  #
    gridded_subset_da = gridded_subset_ds[var]
    station_da = station_ds[var]

    if SHOULD_DEBUG:
        gridded_da.sel(time=DATE, method="nearest").plot()
        plt.plot(station_lon, station_lat, "o")
        plt.show()

        gridded_subset_da.sel(time=DATE, method="nearest").plot()
        plt.plot(station_lon, station_lat, "o")
        plt.show()

        gridded_subset_da.plot.hist(bins=15)
        plt.show()

    for algo_param in algo_params:
        print(f"Now doing {algo_param['name']} bias correction")

        if algo_param["name"] == "Quantile Delta Mapping":
            corrected_da = algo_param["func"](
                gridded_da=gridded_subset_da,
                station_da=station_da[:, 0, 0].drop_vars(["lat", "lon"]),
                should_plot=SHOULD_DEBUG,
            )
        else:
            corrected_da = algo_param["func"](
                gridded_subset_da.sel(time=DATE, method="nearest"),
                station_da=station_da.sel(time=DATE, method="nearest"),
                std_scale=0.1,
                should_plot=SHOULD_DEBUG,
            )

        if SHOULD_DEBUG:
            gridded_subset_slice_da = gridded_subset_da.sel(time=DATE, method="nearest")
            corrected_slice_da = corrected_da.sel(time=DATE, method="nearest")

            plot_min = min(
                [corrected_slice_da.min(), gridded_subset_slice_da.min()]
            ).values
            plot_max = max(
                [corrected_slice_da.max(), gridded_subset_slice_da.max()]
            ).values

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

# %% [markdown]
# ### Plot select dates

# %%
for slice_date in [f"2008-{i:02d}-20" for i in range(7, 12 + 1)]:
    print(slice_date)
    gridded_subset_slice_da = gridded_subset_da.sel(time=slice_date)
    corrected_slice_da = corrected_da.sel(time=slice_date)

    plot_min = min([corrected_slice_da.min(), gridded_subset_slice_da.min()]).values
    plot_max = max([corrected_slice_da.max(), gridded_subset_slice_da.max()]).values

    gridded_subset_slice_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title(f"{slice_date}\n{title}")
    plt.show()

    corrected_slice_da.plot(vmin=plot_min, vmax=plot_max)
    plt.title(
        f"{slice_date}\nCorrected {title}\n{algo_param['name']}\nStation reading: {station_da.sel(time=slice_date).mean().item()}"
    )
    plt.show()

    (corrected_slice_da - gridded_subset_slice_da).plot(cmap="RdYlGn")
    plt.title(
        f"{slice_date}\nDifference between corrected and uncorrected\n{title}\n{algo_param['name']}"
    )
    plt.show()

# %% [markdown]
# ### Scatterplot

# %%
variable_name = "Minimum Temperature"
units = "°C"
plot_offset = 2.5
# variable_name = "Maximum Temperature"
# units = "°C"
# plot_offset = 1
# variable_name = "Precipitation"
# units = "mm/day" #"°C"
# plot_offset = 10

station_aligned_da, corrected_aligned_da = xr.align(
    station_da.mean(dim=["lat", "lon"], skipna=True),
    corrected_da.mean(dim=["lat", "lon"], skipna=True),
    join="inner",
)

corr, pval = sc.pearsonr(station_aligned_da, corrected_aligned_da)
rmse = mean_squared_error(station_aligned_da, corrected_aligned_da, squared=False)

fig, ax = plt.subplots(1, 1)
plot_min = min(station_aligned_da.min(), corrected_aligned_da.min()) - plot_offset
plot_max = max(station_aligned_da.max(), corrected_aligned_da.max()) + plot_offset
ax.plot(
    [plot_min - plot_offset, plot_max + plot_offset],
    [plot_min - plot_offset, plot_max + plot_offset],
    "k-",
)
ax.plot(station_aligned_da, corrected_aligned_da, "o", color="firebrick")
ax.set_aspect(1)
plt.xlabel(f"Station {variable_name} ({units})")
plt.ylabel(f"Corrected {variable_name} ({units})")
plt.title(
    f"Scatterplot of Corrected vs. Station {variable_name}\ncorr: {corr:.3f} pval: {pval:.3f}\nrmse: {rmse:.3f} {units}"
)
plt.xlim(plot_min, plot_max)
plt.ylim(plot_min, plot_max)
plt.show()

# %%
