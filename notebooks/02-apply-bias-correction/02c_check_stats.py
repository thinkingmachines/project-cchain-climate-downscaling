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
import pandas as pd

# Util imports
sys.path.append("../../")

# %%
CITY_NAME = "Davao"

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

# %%

# %%
city_names = [
    "Dagupan",
    "Palayan",
    # "MetroManila",
    "Legazpi",
    "Iloilo",
    "Mandaue",
    "Tacloban",
    "Zamboanga",
    "CagayanDeOro_Lumbia",
    "CagayanDeOro_ElSalvador",
    "Davao",
]

# %%
cities_df = pd.DataFrame(columns=["station", "var", "corr", "rmse"])
for city in city_names:
    city_df = pd.read_parquet(CORRECTED_PATH / f"stats_{city.lower()}.parquet")
    # print(city)
    # print(city_df[["var","corr","rmse"]])
    city_df["station"] = city
    # print(city_df[["station","var","corr","rmse"]])
    cities_df = pd.concat([cities_df, city_df[["station", "var", "corr", "rmse"]]])
cities_df = cities_df.reset_index(drop=True)
cities_df.to_parquet(CORRECTED_PATH / "stats_all_cities.parquet")
cities_df.sort_values("corr")

# %%
