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

# %% [markdown]
# ### Check statistics of bias-corrected output

# %%
PROCESSED_PATH = Path("../../data/02-processed")
CORRECTED_PATH = PROCESSED_PATH / "bias-correction-radial-optimized"

# %%
city_names = [
    "Dagupan",
    "Palayan",
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
cities_df = pd.DataFrame(columns=["station", "var", "method", "corr", "rmse"])
for city in city_names:
    city_df = pd.read_parquet(CORRECTED_PATH / f"stats_{city.lower()}.parquet")
    city_df["station"] = city
    cities_df = pd.concat(
        [cities_df, city_df[["station", "var", "method", "corr", "rmse"]]]
    )
cities_df = cities_df.reset_index(drop=True)
cities_df.to_parquet(CORRECTED_PATH / "stats_all_cities.parquet")
cities_df.sort_values("corr")

# %%
