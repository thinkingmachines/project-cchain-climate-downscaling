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
from tqdm import tqdm
from loguru import logger
import xarray as xr
import requests

# %% [markdown]
# # Download CHIRTS files
#
#
#
# This notebook goes through the process of downloading [CHIRTS daily Tmin and Tmax daily data](https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRTS/.v1.0/.daily/.global/.0p05/index.html?Set-Language=fr) available per year.

# %% [markdown]
# ### Input parameters

# %%
DEST_PATH = Path("../../data/01-raw/chirts")

# %% [markdown]
# ### Setup logging

# %%
LOG_PATH = Path("../../logs/")

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Configure daily rotation for file logging
daily_sink_file_fmt = LOG_PATH / "chirts_{time:YYYY-MM-DD}.log"
logger.add(
    daily_sink_file_fmt,
    rotation="00:00",
    format="{time} {level} {message}",
    level="INFO",
)


# %% [markdown]
# ### Create download function

# %%
def download_file(url, save_path):
    file_name = str(save_path).split("/")[-1]
    logger.info(
        "==========================================================================================="
    )
    logger.info(f"Downloading: {file_name}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 Megabyte
    logger.info(f"Total size: {(total_size/ block_size):.2f} MB")
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(save_path, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        logger.error(f"Downloading {file_name}  failed.")
    else:
        logger.success(f"{file_name} successfully downloaded!")


# %% [markdown]
# ### Download yearly files and subset to PH

# %%
PH_BBOX = (116.5, 4.25, 127, 21.5)
years = [2015]

# %%
for year in years:
    for data_type in ["min", "max"]:
        try:
            file_url = f"https://data.chc.ucsb.edu/products/CHIRTSdaily/v1.0/global_netcdf_p05/T{data_type}/T{data_type}.{year}.nc"
            download_file(file_url, DEST_PATH / "tmp" / f"T{data_type}_{year}.nc")
            # Subset to PH
            ds = xr.open_dataset(DEST_PATH / "tmp" / f"T{data_type}_{year}.nc")
            ds = ds.sel(
                latitude=slice(PH_BBOX[1], PH_BBOX[3]),
                longitude=slice(PH_BBOX[0], PH_BBOX[2]),
            )
            ds.to_netcdf(DEST_PATH / f"CHIRTS_T{data_type}_PH_{year}.nc")
            os.remove(DEST_PATH / "tmp" / f"T{data_type}_{year}.nc")
        except KeyboardInterrupt:
            logger.error("Process interrupted using keyboard.")
            break
