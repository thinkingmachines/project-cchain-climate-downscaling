# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd

# %%
INPUT_PATH = Path("../../data/01-raw")
DEST_PATH = Path("../../data/02-processed")

# %% [markdown]
# ### Read station data

# %%
df_list = []
for data_batch in [
    batch for batch in os.listdir(INPUT_PATH / "station_data") if "A-" in batch
]:
    files_list = [
        fn
        for fn in os.listdir(INPUT_PATH / "station_data" / data_batch)
        if "Data" in fn
    ]
    for fn in files_list:
        df = pd.read_csv(INPUT_PATH / "station_data" / data_batch / fn)
        this_station = fn.split(" Daily")[0]
        print(df.columns)
        df = df.dropna(how="all", axis=0)
        df["date"] = pd.to_datetime(
            df[["YEAR", "MONTH", "DAY"]].astype(int)
        ).dt.strftime("%Y-%m-%d")
        df.columns = [col.lower() for col in df.columns]
        df["station"] = this_station
        df = df.replace(999, np.NaN)
        print(f"{this_station}:{df.isnull().values.ravel().sum()} total missing data")
        df = df.rename(columns={"rr": "rainfall"})
        df["rainfall"] = df["rainfall"].replace(-1.0, 0)
        df = df[
            ["station", "date"]
            + [
                col
                for col in df.columns
                if col not in ["year", "month", "day", "date", "station"]
            ]
        ]
        df_list.append(df)

# %%
alldf = pd.concat(df_list)
alldf

# %%
alldf.to_csv(DEST_PATH / "station_data.csv", index=False)
