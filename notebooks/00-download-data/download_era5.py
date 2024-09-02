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

# %% id="DmkudOVUVITm"
import cdsapi
import os
import sys
from pathlib import Path
import time
from loguru import logger
import itertools
from getpass import getpass

# %% [markdown] id="RTOkZscLPxqJ"
# # Download ERA5 files
#
# This notebook goes through the process of downloading ERA5 Surface data using the Climate Data Store (CDS) API with the following parameters.

# %% [markdown] id="lTR250FsWnCW"
# ### Set up the API Key
# In this section, we install the CDS UID and API Key needed to access the API. To get these, first register for an account [here](https://cds.climate.copernicus.eu/#!/home).

# %%
CDS_API_SECRETS_FILE = Path.home() / ".cdsapirc"
print(f"{CDS_API_SECRETS_FILE} exists: {CDS_API_SECRETS_FILE.is_file()}")

# Set to True to prompt entering API UID and Key again even if file already exists
OVERWRITE_SECRETS = False

# %% colab={"base_uri": "https://localhost:8080/"} id="Anq7h3wUUoth" outputId="c2962ae1-5e4b-4bdd-a0ec-d7cf12f4cfe8"
if not CDS_API_SECRETS_FILE.is_file() or OVERWRITE_SECRETS:
    print(f"Creating new API SECRETS file at {CDS_API_SECRETS_FILE}")
    uid = getpass("Enter your CDS API UID here")
    apikey = getpass("Enter your CDS API Key here")
    key = f"{uid}:{apikey}"

    # Install the API Key
    # https://stackoverflow.com/questions/64304862/using-cdsapi-in-google-colab
    url = "url: https://cds.climate.copernicus.eu/api/v2"
    with open(CDS_API_SECRETS_FILE, "w") as f:
        f.write("\n".join([url, f"key: {key}"]))
    print(f"File created at {CDS_API_SECRETS_FILE}")
else:
    print(f"Using API file at {CDS_API_SECRETS_FILE}")

# %% [markdown]
# ### Create CDS API client object

# %%
c = cdsapi.Client()

# %% [markdown]
# ### Set up logging

# %%
LOG_PATH = Path("../../logs/")

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Configure daily rotation for file logging
daily_sink_file_fmt = str(LOG_PATH / "era5_{time:YYYY-MM-DD}.log")
logger.add(
    daily_sink_file_fmt,
    rotation="00:00",
    format="{time} {level} {message}",
    level="INFO",
)

# %% [markdown] id="IIY9TI-_XFLN"
# ### Define download request parameters for hourly downloads

# %%
## Set user-defined paramters here
PH_BBOX = [
    21.5,  # maxy
    116.5,  # minx
    4.25,  # miny
    127,  # maxx
]

START_YEAR = 2003
END_YEAR = 2022

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_cloud_cover",
    "total_precipitation",
]

# %%
# Set the destination folder
OUTPUT_PATH = Path("../../data/01-raw/era5/")
OUTPUT_PATH.mkdir(exist_ok=True)

# %%
years = [str(x) for x in range(START_YEAR, END_YEAR + 1, 1)]
months = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]

# %% [markdown]
# ### Download monthly files
# Check the logfile in `logs/` to see the progress

# %%
start_time = time.time()

for year, month in itertools.product(years, months):
    try:
        output_filename = f"ERA5_PH_{year}{month}_surface_hourly.nc"
        logger.info(
            f"Downloading for the year {year} and month {month} to filename {output_filename}"
        )

        # Check if output_filename exists
        if (OUTPUT_PATH / output_filename).is_file():
            logger.warning(
                f"{output_filename} already exists in {OUTPUT_PATH}! Skipping request."
            )
            continue

        # Build request parameters
        dataset_short_name = "reanalysis-era5-single-levels"
        request_parameters = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": VARIABLES,
            "year": year,
            "month": month,
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "area": PH_BBOX,
        }
        logger.info("Submitting download request to CDS")
        c.retrieve(
            dataset_short_name, request_parameters, OUTPUT_PATH / output_filename
        )

        # Print out the file size of the output file
        file_stats = os.stat(OUTPUT_PATH / output_filename)
        filesize_MB = file_stats.st_size / (1024 * 1024)
        logger.info(
            f"File download {output_filename} complete! Filesize: {filesize_MB:.2f} Mb"
        )

    # Catch keyboard interrupts
    except KeyboardInterrupt:
        logger.error("Process interrupted using keyboard.")
        break

    # Catch other errors
    except Exception as e:
        logger.error(f"Exception raised: {e}")

end_time = time.time()

# Calculate and print the runtime
runtime_seconds = end_time - start_time
human_readable_runtime = time.strftime("%H:%M:%S", time.gmtime(runtime_seconds))
print("Runtime:", human_readable_runtime)
