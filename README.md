<div align="center">

# Lacuna Fund Climate Downscaling

</div>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>

<br/>
<br/>


# 📜 Description

This repo sets up the local environment to improve existing climate data (temperature and rainfall) of the Lacuna fund linked dataset using ML methods.

<br/>
<br/>


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
