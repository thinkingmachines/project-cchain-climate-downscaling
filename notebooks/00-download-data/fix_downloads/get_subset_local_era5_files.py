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
import pandas as pd
from pathlib import Path
import xarray as xr

# %% [markdown]
# # Get relevant variables from local ERA5 netcdf files
# This notebook obtains a subset of locally downloaded ERA5 files containing only the variables relevant to the downscaling model

# %% [markdown]
# ### Input parameters

# %%
RAW_DIR = Path("/mnt/c/Users/JCPeralta/Downloads/ERA5_converted")
DEST_DIR = Path("../../data/01-raw/era5")

# %%
VARIABLES = ["u10", "v10", "d2m", "t2m", "sp", "tp", "tcc"]

# %% [markdown]
# ### Get subset containing specified variables and save

# %%
for fn in [fn for fn in os.listdir(RAW_DIR) if "idx" not in fn]:
    print(fn)
    output_filename = f"ERA5_PH_{fn.split('_')[2]}.nc"
    ds = xr.open_dataset(RAW_DIR / fn)
    ds = ds[VARIABLES]
    ds.to_netcdf(DEST_DIR / output_filename)

# %% [markdown]
# ### Check if all files are complete

# %%
fns = [fn.split("_")[2] for fn in os.listdir(DEST_DIR)]
complete_dates = pd.date_range(
    start="2003-01-01", end="2022-12-01", freq="MS"
).strftime("%Y%m")
[fn for fn in complete_dates if fn not in fns]
