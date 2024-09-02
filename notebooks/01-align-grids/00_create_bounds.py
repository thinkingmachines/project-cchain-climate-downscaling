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
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
from pathlib import Path

# %% [markdown]
# # Create bounds
# This notebook generates a geojson file from a csv of lat lon bounds and fixes geometries if in case they are not perfect rectangles

# %% [markdown]
# ### Input parameters

# %%
INPUT_PATH = Path("../../data/01-raw/domains")

# %% [markdown]
# ### Create geojson of domain bounds

# %%
df = pd.read_csv(INPUT_PATH / "downscaling_domains.csv")
df

# %%
polygons = []

for index, row in df.iterrows():
    lat0, lat1, lon0, lon1 = row["lat0"], row["lat1"], row["lon0"], row["lon1"]
    polygon = Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
    polygons.append(polygon)

df["geometry"] = polygons
df.index = pd.Index(df.index)
gdf = gpd.GeoDataFrame(df[["city", "geometry"]], geometry="geometry")
gdf

# %%
gdf.to_file(INPUT_PATH / "downscaling_domains.geojson", driver="GeoJSON")

# %% [markdown]
# ### Fix geometries to perfect rectangles

# %%
gdf = gpd.read_file(INPUT_PATH / "downscaling_domains.geojson")
gdf

# %%
perfect_rectangles = []
for index, row in gdf.iterrows():
    rough_polygon = row["geometry"]
    min_x, min_y, max_x, max_y = rough_polygon.bounds
    perfect_polygon = Polygon(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    )
    perfect_rectangles.append(perfect_polygon)
perfect_gdf = gpd.GeoDataFrame(gdf, geometry=perfect_rectangles, crs=gdf.crs)
perfect_gdf

# %%
perfect_gdf.to_file(INPUT_PATH / "downscaling_domains_fixed.geojson", driver="GeoJSON")
