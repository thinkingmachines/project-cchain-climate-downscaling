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
from cmethods import adjust
import matplotlib.pyplot as plt
import xarray as xr

# Util imports
sys.path.append("../../")

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
# ### Quantile mapping
# TODO: Transfer to `climate_downscaling_utils.py`

# %%
def correct_gridded_quantile_mapping(
    gridded_da: xr.DataArray,
    station_da: xr.DataArray,
    method: str = "quantile_delta_mapping",
    n_quantiles: int = 1_000,
    offset: float = 1e-12,
    should_plot: bool = True,
):
    """
    Apply bias correction for a grid within the influence of a station.
    First record the pixelwise deviation to the spatial mean (multiplicative if rainfall and additive otherwise) per timestep.
    Then apply the bias correction.
    Lastly, reapply the deviation to preserve the spatial variability.

    Parameters
    ----------
    gridded_da : xarray DataArray
        Contains the gridded variables.

    station_da : xarray DataArray
        Contains the variables for a single station.

    method : string
        Bias correction method under the adjust function of the cmethods package.
        Default is "quantile_delta_mapping".

    n_quantiles : int
        Optional, number of quantiles for "quantile_delta_mapping".
        Default is 1_000.

    offset : float
        Numerical offset for calculating the deviation.
        Default is 1 x 10^(-12).

    should_plot : bool
        If True, plots the frequency distributions.

    Returns
    -------
    xarray DataArray
    """
    bias_correction_kind = "*" if gridded_da.name == "precip" else "+"

    # get the spatial mean
    gridded_mean_da = gridded_da.mean(dim=["lat", "lon"], skipna=True)

    if gridded_da.name == "precip":
        gridded_deviation_da = gridded_da / max(gridded_mean_da, offset)
    else:
        gridded_deviation_da = gridded_da - gridded_mean_da

    # observation (obs) is the station_da timeseries
    # historical simulation (simh) is gridded_da since it has the bias
    # predicted simulation (simp) is also gridded_da since we are correcting that data
    corrected_da = adjust(
        method=method,
        obs=station_da,
        simh=gridded_mean_da,
        simp=gridded_mean_da,
        n_quantiles=n_quantiles,
        kind=bias_correction_kind,
    )

    corrected_3d_da = corrected_da.expand_dims(
        dim=dict(
            lat=gridded_da["lat"].shape[0],
            lon=gridded_da["lon"].shape[0],
        )
    ).assign_coords(
        dict(
            lat=gridded_da["lat"],
            lon=gridded_da["lon"],
        )
    )

    if gridded_da.name == "precip":
        corrected_variability_da = corrected_3d_da * gridded_deviation_da
    else:
        corrected_variability_da = corrected_3d_da + gridded_deviation_da

    return corrected_variability_da


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
        func=correct_gridded_quantile_mapping,
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

    gridded_da = gridded_ds[var]  # .sel(time=DATE, method="nearest")
    gridded_subset_da = gridded_subset_ds[var]  # .sel(time=DATE, method="nearest")

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

        if algo_param["name"] == "Quantile Delta Mapping":
            corrected_da = algo_param["func"](
                gridded_subset_da,
                station_da=station_ds[var],  # .sel(time=DATE, method="nearest"),
                should_plot=SHOULD_DEBUG,
            )
    else:
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
gridded_da = gridded_subset_da
station_da = station_ds[var][:, 0, 0].drop_vars(
    ["lat", "lon"]
)  # .sel(time=DATE, method="nearest")
method = "quantile_delta_mapping"
n_quantiles = 1_000
offset = 1e-12

bias_correction_kind = "*" if gridded_da.name == "precip" else "+"

# get the spatial mean
gridded_mean_da = gridded_da.mean(dim=["lat", "lon"], skipna=True)

if gridded_da.name == "precip":
    gridded_deviation_da = gridded_da / max(gridded_mean_da, offset)
else:
    gridded_deviation_da = gridded_da - gridded_mean_da

# align DataArrays to match dimensions
station_aligned_da, gridded_aligned_da = xr.align(
    station_da, gridded_mean_da, join="inner"
)

# observation (obs) is the station_da timeseries
# historical simulation (simh) is gridded_da since it has the bias
# predicted simulation (simp) is also gridded_da since we are correcting that data
corrected_ds = adjust(
    method=method,
    obs=station_aligned_da,
    simh=gridded_aligned_da,
    simp=gridded_aligned_da,
    n_quantiles=n_quantiles,
    kind=bias_correction_kind,
)

# nlat = gridded_da["lat"].shape[0]
# nlon = gridded_da["lon"].shape[0]
# ntime = gridded_aligned_da.values.shape[0]
# corrected_3d_da = xr.DataArray(
#     nlat*[
#         nlon*[
#             corrected_ds[gridded_da.name].values[:,0,0]
#         ]
#     ],
#     dims=dict(
#         lat=gridded_da["lat"],
#         lon=gridded_da["lon"],
#         time=corrected_ds["time"],
#     ),
#     coords=dict(
#         lat=gridded_da["lat"],
#         lon=gridded_da["lon"],
#         time=corrected_ds["time"],
#     )
# )

corrected_3d_da = (
    corrected_ds
    # .drop_dims(
    #     [
    #         "lat",
    #         "lon",
    #     ]
    # )
    .expand_dims(
        dim=dict(
            lat=gridded_da["lat"].shape[0],
            lon=gridded_da["lon"].shape[0],
        ),
        # create_index_for_new_dim=False,
    ).transpose("time", "lat", "lon")
    # .assign_coords(
    #     dict(
    #         lat=gridded_da["lat"],
    #         lon=gridded_da["lon"],
    #     )
    # )
)

if gridded_da.name == "precip":
    corrected_variability_da = corrected_3d_da * gridded_deviation_da
else:
    corrected_variability_da = corrected_3d_da + gridded_deviation_da

# %%
corrected_variability_da["tmin"][0].plot()

# %%
gridded_deviation_da[0].plot()

# %%
station_da.sel(time="2008-07-01").values

# %%
corrected_variability_da.sel(time="2008-07-01")["tmin"].plot()

# %%
gridded_subset_ds.sel(time="2008-07-01")["tmin"].plot()

# %%
(corrected_variability_da.sel(time="2008-07-01") - gridded_da.sel(time="2008-07-01"))[
    "tmin"
].plot()

# %%
station_aligned_da[:, 0, 0].drop_vars(["lat", "lon"])  # .drop_indexes(["lat","lon"])

# %%
gridded_aligned_da

# %%
corrected_ds.reindex(
    dict(
        lat=gridded_da["lat"],
        lon=gridded_da["lon"],
    ),
    method="ffill",
)["tmin"][0].plot()

# %%
corrected_variability_da

# %%
gridded_subset_ds["tmin"][0].plot()

# %%
corrected_variability_da["tmin"][:, :, 0].plot()

# %%
(corrected_variability_da["tmin"][:, :, 0] - gridded_subset_ds["tmin"][0]).plot()

# %%
gridded_subset_ds["tmin"][0].mean(dim=["lat", "lon"])

# %%
corrected_3d_da["tmin"][:, :, 0]

# %%
corrected_3d_da["tmin"][:, :, 0].plot()

# %%
gridded_deviation_da[0].plot()

# %%
station_da

# %%
gridded_mean_da

# %%
a, b = xr.align(station_da, gridded_mean_da, join="inner")

# %%
b

# %%
