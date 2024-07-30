from typing import Any, List

from cmethods import adjust
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import xarray as xr


def correct_gridded_liu(
    gridded_da: xr.DataArray,
    station_da: xr.DataArray,
    std_scale: float = 0.1,
    should_plot: bool = False,
) -> xr.DataArray:
    """
    Apply bias correction based on Liu et al. (2019) for a grid within the influence of a station.
    Assume the station standard deviation is a fraction of the mean.
    For the computation of k (ratio), n was used instead of n_max.

    Parameters
    ----------
    gridded_da : xarray DataArray
        Contains the gridded variables.

    station_da : xarray DataArray
        Contains the variables for a single station.

    std_scale : float or int
        Scaling factor on the station mean to get the station standard deviation.

    should_plot : bool
        If True, plots the frequency distributions.
        Default is False.

    Returns
    -------
    xarray DataArray
    """
    gridded_arr = gridded_da.values
    gridded_mean = np.nanmean(gridded_arr)
    gridded_std = max(np.nanstd(gridded_arr), 1e-12)  # avoid 0 std

    station_mean = station_da.values[0]
    station_std = max(std_scale * station_mean, 1e-12)  # avoid 0 std

    # compute for the ratio k by assuming both datasets follow a normal distribution
    n = np.linspace(station_mean - 4 * station_std, station_mean + 4 * station_std, 100)
    ratio = norm.pdf(n, loc=gridded_mean, scale=gridded_std) / norm.pdf(
        n, loc=station_mean, scale=station_std
    )
    ratio_da = custom_replace(gridded_da, n, ratio)

    # compute for the corrected gridded data
    corrected_da = (
        station_std
        * np.sqrt(
            2 * np.log(ratio_da * gridded_std / station_std)
            + ((gridded_da - gridded_mean) / gridded_std) ** 2
        )
        + station_mean
    )

    # visualize the frequency distributions
    if should_plot:
        plt.plot(
            n, norm.pdf(n, loc=station_mean, scale=station_std), "o", label="station"
        )
        plt.plot(
            n,
            norm.pdf(n, loc=gridded_mean, scale=gridded_std),
            "o",
            label="gridded",
        )
        plt.legend()
        plt.show()

    return corrected_da


def correct_gridded_quantile_mapping(
    gridded_da: xr.DataArray,
    station_da: xr.DataArray,
    method: str = "quantile_delta_mapping",
    n_quantiles: int = 1_000,
    offset: float = 1e-12,
    should_plot: bool = False,
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
        gridded_deviation_da = gridded_da / (
            gridded_mean_da.where(abs(gridded_mean_da) > offset, other=offset)
        )
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

    corrected_3d_da = corrected_ds.expand_dims(
        dim=dict(
            lat=gridded_da["lat"].shape[0],
            lon=gridded_da["lon"].shape[0],
        ),
    ).transpose("time", "lat", "lon")[gridded_da.name]

    if gridded_da.name == "precip":
        corrected_variability_da = corrected_3d_da * gridded_deviation_da
    else:
        corrected_variability_da = corrected_3d_da + gridded_deviation_da

    return corrected_variability_da


def correct_gridded_zscore(
    gridded_da: xr.DataArray,
    station_da: xr.DataArray,
    std_scale: float | int = 10,
    should_plot: bool = False,
) -> xr.DataArray:
    """
    Apply bias correction based on the z-score formula for a grid within the influence of a station.
    Assume the station standard deviation is a fraction of the mean.
    z-score = (value - mean)/std

    Parameters
    ----------
    gridded_da : xarray DataArray
        Contains the gridded variables.

    station_da : xarray DataArray
        Contains the variables for a single station.

    std_scale : float or int
        Scaling factor on the station mean to get the station standard deviation.

    should_plot : bool
        If True, plots the frequency distributions.
        Default is False.

    Returns
    -------
    xarray DataArray
    """
    gridded_arr = gridded_da.values
    gridded_mean = np.nanmean(gridded_arr)
    gridded_std = max(np.nanstd(gridded_arr), 1e-12)  # avoid 0 std

    station_mean = station_da.values[0]
    station_std = max(std_scale * station_mean, 1e-12)  # avoid 0 std

    # compute for the corrected gridded data
    corrected_da = (
        (gridded_da - gridded_mean) / gridded_std
    ) * station_std + station_mean

    # visualize the frequency distributions
    if should_plot:
        n = np.linspace(
            station_mean - 4 * station_std, station_mean + 4 * station_std, 100
        )
        plt.plot(
            n, norm.pdf(n, loc=station_mean, scale=station_std), "o", label="station"
        )
        plt.plot(
            n,
            norm.pdf(n, loc=gridded_mean, scale=gridded_std),
            "o",
            label="gridded",
        )
        plt.legend()
        plt.show()

    return corrected_da


def custom_replace(
    da: xr.DataArray,
    to_replace: List[Any],
    value: List[Any],
) -> xr.DataArray:
    """
    Replace a set of values in an xarray DataArray
    From https://github.com/pydata/xarray/issues/6377#issue-1173497454

    Parameters
    ----------
    da : xarray DataArray

    to_replace : list
        List of values in da to be replaced.

    value : list
        List of values replacing to_replace.

    Returns
    -------
    xarray DataArray
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
