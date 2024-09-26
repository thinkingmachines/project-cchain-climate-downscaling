<div align="center">

# Project CCHAIN - Deep learning climate downscaling model

</div>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>

<br/>
<br/>


# 📜 Description

The [Project CCHAIN dataset](https://thinkingmachines.github.io/project-cchain) contains daily climate variables over 12 Philippine cities that were designed to be used in health and other applied research. However, the values are from coarse global gridded data and are not resolved for barangay level (village level in the Philippines). 

Computational methods called **climate downscaling** addresses this issue by using simulations or statistical processes to increase the resolution of coarse climate data. A technique called *dynamical climate downscaling* uses physics equations simulating large-scale atmospheric systems to approximate a desired finer resolution. However, this method takes large computational resources, both on hardware and runtime. Recent advances in the past decade has allowed machine learning models to help push downscaling forward by detecting and transferring patterns from available high resolution data (e.g. from weather stations, radar) to correct coarse resolution data. The end result is also higher resolution climate data but produced significantly faster and requires less resources. 

Open source machine learning models that downscale climate data have been developed by leading institutions in developed nations. One such model is [dl4ds](https://www.cambridge.org/core/journals/environmental-data-science/article/dl4dsdeep-learning-for-empirical-downscaling/5D0623A860C6082FD650D704A50BEF3D),  a python module that implements range of architectures for downscaling gridded data with deep neural networks (Gonzales 2023).

![Sample](assets/tmax_CagayanDeOro_2016-01-20_comparison.png?raw=true "Sample downscaled maximum temperature for Cagayan De Oro City")

With support from the Lacuna Fund, we are able to create this code that allows us to improve the resolution of the currently provided temperature and rainfall data to bring it down to the local level. It is our hope that local developers can use, contribute and grow this code base to add more capabilities that may be useful to our stakeholders


<br/>
<br/>

# ⚠️ For data users: Using the provided output
The model yielded minimum temperature, maximum temperature and rainfall with enhanced resolution from the reanalysis scale (0.25°) to local scale (0.02°). However, given the uncertainties/biases in the magnitude of the downscaled temperature and rainfall, we advise users not to treat the output the way they would treat ground-measured data (e.g. station data) but focus on its bulk statistical characteristics (distribution, timing, spatial pattern) instead.

While we provide the full downscaled output as gridded netcdf files [here](https://drive.google.com/drive/u/0/folders/1mXaFEhMYZnLzUCX3RciK5JEmguf_UHdd) for all the 12 cities, only those variables that passed our quality checks (QC) are included in the extracted data. These are the following:

| **City**       | tmin | tmax | pr |
|----------------|------|------|----|
| Palayan        | ✓    | ✓    | ✓  |
| Dagupan        | ✓    | ✓    | ✓  |
| Davao          |      | ✓    | ✓  |
| Cagayan De Oro |      | ✓    | ✓  |
| Iloilo         | ✓    | ✓    |    |
| Legazpi        |      |      | ✓  |
| Mandaue        | ✓    |      | ✓  |
| Muntinlupa     |      | ✓    | ✓  |
| Navotas        |      | ✓    | ✓  |
| Mandaluyong    |      | ✓    | ✓  |
| Tacloban       | ✓    | ✓    | ✓  |
| Zamboanga      | ✓    | ✓    | ✓  |

You may view a more detailed showcase of results here in these [slides](https://docs.google.com/presentation/d/1y8mAa07aC7loeY2e5Oqicy98hxFag6kUhyAp8Qp59U4/). If you are uncertain, consider using the coarse data provided in the [climate_atmosphere](https://dbdocs.io/lacuna-fund-project-team/Project-CCHAIN?table=climate_atmosphere&schema=public&view=table_structure) table instead.

# ⚙️ Local Setup for Development

This repo assumes the use of [conda](https://docs.conda.io/en/latest/) for simplicity in installing GDAL.


## Requirements

1. Python 3.10
2. make
3. conda


## 🐍 One-time Set-up
Run this the very first time you are setting-up the project on a machine to set-up a local Python environment for this project.

1. Install [miniforge](https://github.com/conda-forge/miniforge) for your environment if you don't have it yet.
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

2. Create a local conda env `climate-downscaling` as named in `environment.yml` and activate it. This will create a conda env folder in your project directory.
```
make env
conda activate climate-downscaling
```

3. Run the one-time set-up make command.
```
make setup
```

## 🐍 Testing
To run automated tests, simply run `make test`.

## 📦 Dependencies

Over the course of development, you will likely introduce new library dependencies. This repo uses [uv](https://github.com/astral-sh/uv) to manage the python dependencies.

There are two main files involved:
* `pyproject.toml` - contains project information and high level requirements; this is what we should edit when adding/removing libraries
* `requirements.txt` - contains exact list of python libraries (including depdenencies of the main libraries) your environment needs to follow to run the repo code; compiled from `pyproject.toml`


When you add new python libs, please do the ff:

1. Add the library to the `pyproject.toml` file in the `dependencies` list under the `[project]` section. You may optionally pin the version if you need a particular version of the library.

2. Run `make requirements` to compile a new version of the `requirements.txt` file and update your python env.

3. Commit both the `pyproject.toml` and `requirements.txt` files so other devs can get the updated list of project requirements.

Note: When you are the one updating your python env to follow library changes from other devs (reflected through an updated `requirements.txt` file), simply run `uv pip sync requirements.txt`
