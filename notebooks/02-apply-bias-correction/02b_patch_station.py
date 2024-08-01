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
import xarray as xr

# Util imports
sys.path.append("../../")

# %%
PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction"

# %%
city_names = [
    "CagayanDeOro_Lumbia",
    "CagayanDeOro_ElSalvador",
]

ds = xr.concat(
    [
        xr.open_dataset(
            CORRECTED_PATH / f"corrected_subset_{city_names[0].lower()}.nc"
        ),
        xr.open_dataset(
            CORRECTED_PATH / f"corrected_subset_{city_names[1].lower()}.nc"
        ),
    ],
    dim="time",
)
ds.to_netcdf(
    CORRECTED_PATH / f"corrected_subset_{city_names[0].split('_')[0].lower()}.nc",
    engine="scipy",
)
ds

# %%
