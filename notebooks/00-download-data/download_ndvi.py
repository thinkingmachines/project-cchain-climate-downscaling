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
import os
import sys
from pathlib import Path
import time
from loguru import logger
import earthaccess
import pandas as pd
import xarray as xr

# %% [markdown]
# # Download NDVI files
#
# This notebook goes through the process of downloading NDVI data in particular, the [MCD19A3CMG v061](https://lpdaac.usgs.gov/products/mcd19a3cmgv061/) data product, using module `earthaccess`.

# %% [markdown]
# ### Set up credentials
# Key in username and password when prompted

# %%
earthaccess.login(persist=True)

# %% [markdown]
# ### Input parameters
# `BATCH` is used to mark parallel downloads and make sure they dont get mixed up

# %%
DEST_PATH = Path("../../../data/01-raw/ndvi")
BATCH = 1

# %%
PH_BBOX = (116.5, 4.25, 127, 21.5)
start_year = 2003
end_year = 2006

# %% [markdown]
# ### Setup logging

# %%
LOG_PATH = Path("../../logs/")

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Configure daily rotation for file logging
daily_sink_file_fmt = str(LOG_PATH / "era5_{time:YYYY-MM-DD}_batch") + f"{BATCH}.log"
logger.add(
    daily_sink_file_fmt,
    rotation="00:00",
    format="{time} {level} {message}",
    level="INFO",
)


# %% [markdown]
# ### Generate monthly bounding dates

# %%
def generate_monthly_date_pairs(start_year, end_year):
    date_pairs = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = pd.Timestamp(year, month, 1)
            end_date = start_date + pd.offsets.MonthEnd(0)
            date_pairs.append(
                (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            )

    return date_pairs


# %%
date_pairs = generate_monthly_date_pairs(start_year, end_year)
date_pairs

# %% [markdown]
# ### Create monthly batch dump folder

# %%
os.makedirs(DEST_PATH / f"hdfs{BATCH}", exist_ok=True)

# %%
# Delete files if directory exists
for fn in os.listdir(DEST_PATH / f"hdfs{BATCH}"):
    os.remove(DEST_PATH / f"hdfs{BATCH}" / fn)

# %% [markdown]
# ### Download daily files and save as single monthly file
# Check the logfile in `logs/` to see the progress

# %%
VARIABLES = ["NDVI_gapfill", "EVI"]  # EVI optional to keep

# %%
for months in date_pairs:
    start_time = time.time()
    month_fn = "".join(months[0].split("-")[:-1])
    output_filename = f"NDVI_PH_{month_fn}.nc"
    logger.info(
        "==========================================================================================="
    )

    logger.info(
        f"Downloading data for the month {months[0]} to filename {output_filename}"
    )

    if (DEST_PATH / output_filename).is_file():
        logger.warning(
            f"{output_filename} already exists in {DEST_PATH}! Skipping request."
        )
        continue

    results = earthaccess.search_data(
        short_name="MCD19A3CMG",
        cloud_hosted=True,
        bounding_box=PH_BBOX,
        temporal=months,
    )

    files = earthaccess.download(results, DEST_PATH / f"hdfs{BATCH}")

    # Stop and check if folder has expected number of files
    try:
        assert len(results) == len(os.listdir(DEST_PATH / f"hdfs{BATCH}"))
    except Exception as e:
        logger.error(f"Exception raised: {e}")
        logger.error(
            f"{month_fn } has incomplete downloaded data! Please check and rerun, skipping..."
        )
        for fn in os.listdir(DEST_PATH / f"hdfs{BATCH}"):
            os.remove(DEST_PATH / f"hdfs{BATCH}" / fn)
        continue

    logger.info(f"Processing daily data to create {output_filename}...")

    # download daily files
    ds_list = []
    for fn in os.listdir(DEST_PATH / f"hdfs{BATCH}"):
        ds = xr.open_dataset(DEST_PATH / f"hdfs{BATCH}" / fn, engine="rasterio")
        # subset to bounding box and variable list
        ds = ds[VARIABLES]
        ds = ds.sel(y=slice(PH_BBOX[3], PH_BBOX[1]), x=slice(PH_BBOX[0], PH_BBOX[2]))
        # add time dimension
        file_dt = pd.to_datetime(
            f"{ds.attrs['EQUATORCROSSINGDATE.1']} {ds.attrs['EQUATORCROSSINGTIME.1'].split('.')[0]}"
        ).to_datetime64()
        ds = ds.assign_coords(time=file_dt)
        ds = ds.expand_dims(dim="time")
        ds_list.append(ds)

    # concatenate daily datasets and save as netcdf
    month_ds = xr.concat(ds_list, dim="time")
    month_ds.to_netcdf(DEST_PATH / output_filename)

    end_time = time.time()
    runtime_seconds = end_time - start_time
    human_readable_runtime = time.strftime("%H:%M:%S", time.gmtime(runtime_seconds))

    logger.success(f"File {output_filename} done in {human_readable_runtime}")
    # clean download dump folder
    for fn in os.listdir(DEST_PATH / f"hdfs{BATCH}"):
        os.remove(DEST_PATH / f"hdfs{BATCH}" / fn)

# %%
