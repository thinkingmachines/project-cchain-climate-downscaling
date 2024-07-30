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
CITY_NAME = "Dagupan"
DATE = "2008-07-01"  # sample date for debugging
YEARS = [2007, 2008, 2009, 2016, 2017, 2018]
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
# - `plot_offset`: margin for x and y bounds

# %%
variable_params = dict(
    tmin="CHIRTS minimum temperature",
    tmax="CHIRTS maximum temperature",
    precip="CHIRPS precipitation",
)

algo_params = [
    dict(
        name="Linear Scaling",
        func=cd.correct_gridded_cmethods,
        method="linear_scaling",
        group="time.month",
    ),
    dict(
        name="Variance Scaling",
        func=cd.correct_gridded_cmethods,
        method="variance_scaling",
        group="time.month",
    ),
    dict(
        name="Delta Method",
        func=cd.correct_gridded_cmethods,
        method="delta_method",
        group="time.month",
    ),
    dict(
        name="Quantile Mapping",
        func=cd.correct_gridded_cmethods,
        method="quantile_mapping",
        n_quantiles=1_000,
    ),
    # dict(
    #     name="Detrended Quantile Mapping", # got numpy error
    #     func=cd.correct_gridded_cmethods,
    #     method="detrended_quantile_mapping",
    #     n_quantiles=1_000,
    # ),
    dict(
        name="Quantile Delta Mapping",
        func=cd.correct_gridded_cmethods,
        method="quantile_delta_mapping",
        n_quantiles=1_000,
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

# refer to defaults
for algo_param in algo_params:
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
gridded_ds.isel(time=gridded_ds.time.dt.year.isin(YEARS))

# %%
stats_df = pd.DataFrame(columns=["var", "method", "corr", "pval", "rmse"])
stats_df

# %%
stats_list = []
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

    for algo_param in algo_params:
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
                method=algo_param["method"],
                n_quantiles=algo_param["n_quantiles"],
                group=algo_param["group"],
                should_plot=SHOULD_DEBUG,
            )

        station_aligned_da, corrected_aligned_da = xr.align(
            station_da.mean(dim=["lat", "lon"], skipna=True).dropna(dim="time"),
            corrected_da.mean(dim=["lat", "lon"], skipna=True).dropna(dim="time"),
            join="inner",
        )

        corr, pval = sc.pearsonr(station_aligned_da, corrected_aligned_da)
        rmse = mean_squared_error(
            station_aligned_da, corrected_aligned_da, squared=False
        )

        stats_list.append(
            dict(var=var, method=algo_param["method"], corr=corr, pval=pval, rmse=rmse)
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
stats_df = pd.DataFrame(stats_list)

# %%
stats_df[["var", "method", "corr", "rmse"]]

# %%
stats_df[["var", "method", "corr", "rmse"]].plot(marker="o", linestyle="")

# %%
color_dict = dict(
    tmin="orange",
    tmax="firebrick",
    precip="dodgerblue",
)

method_dict = dict(
    linear_scaling="o",
    variance_scaling="*",
    delta_method="^",
    quantile_mapping="s",
    quantile_delta_mapping="D",
)

for var in stats_df["var"].unique():
    for method in stats_df["method"].unique():
        corr_n = stats_df.where(
            (stats_df["var"] == var) & (stats_df["method"] == method)
        ).dropna()["corr"]
        rmse_n = stats_df.where(
            (stats_df["var"] == var) & (stats_df["method"] == method)
        ).dropna()["rmse"]
        plt.plot(
            corr_n,
            rmse_n,
            marker=method_dict[method],
            color=color_dict[var],
            label=f"{var}-{method}",
            alpha=0.5,
        )
# plt.legend()
plt.show()

# %%
corr_n

# %%
stats_df.where((stats_df["var"] == var) & (stats_df["method"] == method)).head(1)[
    "corr"
]

# %% [markdown]
# ### Plot select dates

# %%
# for slice_date in [f"2008-{i:02d}-20" for i in range(7, 12 + 1)]:
#     print(slice_date)
#     gridded_subset_slice_da = gridded_subset_da.sel(time=slice_date)
#     corrected_slice_da = corrected_da.sel(time=slice_date)

#     plot_min = min([corrected_slice_da.min(), gridded_subset_slice_da.min()]).values
#     plot_max = max([corrected_slice_da.max(), gridded_subset_slice_da.max()]).values

#     gridded_subset_slice_da.plot(vmin=plot_min, vmax=plot_max)
#     plt.title(f"{slice_date}\n{title}")
#     plt.show()

#     corrected_slice_da.plot(vmin=plot_min, vmax=plot_max)
#     plt.title(
#         f"{slice_date}\nCorrected {title}\n{algo_param['name']}\nStation reading: {station_da.sel(time=slice_date).mean().item()}"
#     )
#     plt.show()

#     (corrected_slice_da - gridded_subset_slice_da).plot(cmap="RdYlGn")
#     plt.title(
#         f"{slice_date}\nDifference between corrected and uncorrected\n{title}\n{algo_param['name']}"
#     )
#     plt.show()

# %% [markdown]
# ### Scatterplot

# %%
# station_aligned_da, corrected_aligned_da = xr.align(
#     station_da.mean(dim=["lat", "lon"], skipna=True).dropna(dim="time"),
#     corrected_da.mean(dim=["lat", "lon"], skipna=True).dropna(dim="time"),
#     join="inner",
# )

# corr, pval = sc.pearsonr(station_aligned_da, corrected_aligned_da)
# rmse = mean_squared_error(station_aligned_da, corrected_aligned_da, squared=False)

# fig, ax = plt.subplots(1, 1)
# plot_min = min(station_aligned_da.min(), corrected_aligned_da.min()) - scatterplot_params[var]["plot_offset"]
# plot_max = max(station_aligned_da.max(), corrected_aligned_da.max()) + scatterplot_params[var]["plot_offset"]
# ax.plot(
#     [plot_min - scatterplot_params[var]["plot_offset"], plot_max + scatterplot_params[var]["plot_offset"]],
#     [plot_min - scatterplot_params[var]["plot_offset"], plot_max + scatterplot_params[var]["plot_offset"]],
#     "k-",
# )
# ax.plot(station_aligned_da, corrected_aligned_da, "o", color=scatterplot_params[var]["color"])
# ax.set_aspect(1)
# plt.xlabel(f"Station {scatterplot_params[var]['variable_name']} ({scatterplot_params[var]['units']})")
# plt.ylabel(f"Corrected {scatterplot_params[var]['variable_name']} ({scatterplot_params[var]['units']})")
# plt.title(
#     f"Scatterplot of Corrected vs. Station {scatterplot_params[var]['variable_name']}\ncorr: {corr:.3f} pval: {pval:.3f}\nrmse: {rmse:.3f} {scatterplot_params[var]['units']}"
# )
# plt.xlim(plot_min, plot_max)
# plt.ylim(plot_min, plot_max)
# plt.show()

# %%
# df_time = station_aligned_da["time"].to_pandas().reset_index()
# df_time["year_month"] = pd.to_datetime(df_time["time"], format="%Y-%m-%d").dt.strftime("%Y-%m")
# df_time["year_month"].unique()

# %%
