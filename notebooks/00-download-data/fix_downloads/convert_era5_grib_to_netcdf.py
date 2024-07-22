# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ERA5 GRIB to NETCDF Conversion
# This notebook converts the ERA5 downloaded files (in GRIB format) to the appropriate netCDF format

# %%
import xarray as xr
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import netCDF4 as nc
from loguru import logger
from tqdm import tqdm
import zipfile

# %% [markdown]
# ## Pre-requisite Install: ecCodes
# To install ecCodes,  run `conda install -c conda-forge eccodes` to install the binaries AND update the `requirements.in` file to build the python bindings 
#
# For more detailed instructions on how to build from the source files, see [How to Install Eccodes on Ubuntu](https://gist.github.com/MHBalsmeier/a01ad4e07ecf467c90fad2ac7719844a)

# %% [markdown]
# ## Set up logging

# %%
LOG_PATH = Path("../../logs/")

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Configure daily rotation for file logging
daily_sink_file_fmt = str(
    LOG_PATH / "ERA5_Surface_grib_to_nc_conversion_{time:YYYY-MM-DD}.log"
)
logger.add(
    daily_sink_file_fmt,
    rotation="00:00",
    format="{time} {level} {message}",
    level="TRACE",
)


# %%
# Helper function for writing subprocess outputs as trace logs
def log_subprocess_output_trace(pipe):
    """Logs output from subprocess pipe as trace messages
    In the current logging setup, 'trace' messages are written only
    to the log file (level="TRACE") and not sys.stderr / notebook output
    (level="INFO")
    """
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        if line.isspace():
            continue

        logger.trace(f'{str(line,"utf-8").strip()}')


# %% [markdown]
# ## Convert from GRIB to netcdf

# %%
# Helper function for testing if file can be successfully read as netcdf
def is_valid_nc(file):
    try:
        nc.Dataset(file)
        return True
    except Exception as e:
        logger.error(f"Exception raised: {e}")
        return False


# %%
# Function to extract files from zip archive
def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)


# %%
# Set the destination folder
RAW_DIR = Path("/mnt/c/Users/JCPeralta/Downloads")
INPUT_GRIB_FOLDER = Path("/mnt/c/Users/JCPeralta/Downloads/ERA5")
OUTPUT_NC_FOLDER = Path("/mnt/c/Users/JCPeralta/Downloads/ERA5_converted")
OUTPUT_NC_FOLDER.mkdir(exist_ok=True)

# %%
year = 2004
zip_file = RAW_DIR / f"ERA5_{year}.zip"
extract_zip(zip_file, INPUT_GRIB_FOLDER)

# %%
# Get list of downloaded surface files
era5_surface_grib_files = sorted(
    list(INPUT_GRIB_FOLDER.glob("**/*_surface_hourly.nc")), reverse=True
)
era5_surface_grib_files[:3], len(era5_surface_grib_files)

# %%
for fn in os.listdir(INPUT_GRIB_FOLDER):
    os.remove(INPUT_GRIB_FOLDER / fn)

# %%
for year in [2019]:  # range(2018,2022):
    zip_file = RAW_DIR / f"ERA5_{year}.zip"
    extract_zip(zip_file, INPUT_GRIB_FOLDER)

    # Get list of downloaded surface files
    era5_surface_grib_files = sorted(
        list(INPUT_GRIB_FOLDER.glob("**/*_surface_hourly.nc")), reverse=True
    )
    era5_surface_grib_files[:3], len(era5_surface_grib_files)

    for f in tqdm(era5_surface_grib_files):
        try:
            # Check if input file is a valid nc file
            logger.info(f"Input file {f.name} is valid netcdf file: {is_valid_nc(f)}")
            if is_valid_nc(f):
                logger.info("Input file {f} is already valid! skipping request.")
                continue

            output_filename = f.name
            output_filepath = OUTPUT_NC_FOLDER / output_filename
            logger.info(
                f"Converting {f.name} from GRIB to NC and saving to {OUTPUT_NC_FOLDER}"
            )

            # Check if output_filename exists
            if (OUTPUT_NC_FOLDER / output_filename).is_file():
                logger.warning(f"{output_filepath} already exists! Skipping request.")
                continue

            # Specifying conversion command
            # -S param is an option set by ECMWF
            grib2netcf_command_str = (
                f"grib_to_netcdf -S param -o {str(output_filepath)} {str(f)}"
            )
            logger.info(f"Running command: {grib2netcf_command_str}")

            # Run conversion process and log messages to file
            conversion_process = subprocess.Popen(
                grib2netcf_command_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with conversion_process.stdout:
                log_subprocess_output_trace(conversion_process.stdout)
            exitcode = conversion_process.wait()  # waits for process to finish

            logger.info(
                f"Output file {output_filepath} is valid netcdf file: {is_valid_nc(output_filepath)}"
            )
            if is_valid_nc(output_filepath):
                logger.info(f"Successfuly converted {output_filepath}.")
            else:
                logger.warning(f"Conversion {output_filepath} not successful.")

        # Catch keyboard interrupts
        except KeyboardInterrupt:
            # Kill process if running
            try:
                conversion_process.kill()
            except Exception as e:
                logger.error(f"Exception raised: {e}")
            logger.error("Process interrupted using keyboard.")
            break

        # Catch other errors
        except Exception as e:
            logger.error(f"Exception raised: {e}")

    for fn in os.listdir(INPUT_GRIB_FOLDER):
        os.remove(INPUT_GRIB_FOLDER / fn)

# %% [markdown]
# ## Plot most recent output file

# %%
existing_output_files = list(OUTPUT_NC_FOLDER.glob("**/*"))
existing_output_files = [(f, f.lstat().st_mtime) for f in existing_output_files]
existing_output_files

# %%
most_recent_file = max(existing_output_files, key=lambda x: x[1])[0]
print(most_recent_file)

# %%
ds = xr.open_dataset(most_recent_file)
ds

# %%
# crs argument in ax.set_extent
lon_min = 116.5
lon_max = 127
lat_min = 4.25
lat_max = 21.5

VARIABLE_NAME = "t2m"

# %%
projection = ccrs.Mercator()
crs = ccrs.PlateCarree()
plt.figure(dpi=150)
ax = plt.axes(projection=projection, frameon=True)

# Draw gridlines in degrees over Mercator map
gl = ax.gridlines(
    crs=crs, draw_labels=True, linewidth=0.6, color="gray", alpha=0.5, linestyle="-."
)
gl.xlabel_style = {"size": 7}
gl.ylabel_style = {"size": 7}

# To plot borders and coastlines, we can use cartopy feature
ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

cbar_kwargs = {
    "orientation": "horizontal",
    "shrink": 0.8,
    "pad": 0.05,
    "label": "2 Metre Temperature [K]",
}
ds[VARIABLE_NAME].isel(time=1).plot.imshow(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    cbar_kwargs=cbar_kwargs,
    levels=21,
)

# crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
