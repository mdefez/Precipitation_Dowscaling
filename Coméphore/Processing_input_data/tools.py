# This file stores useful functions concerning the Coméphore dataset

from turtle import down
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Example of path to projected data
path_projected = f"../../../downscaling/mdefez/Comephore/Projected_data/test/9829/2019/COMEPHORE_2019_2/2019/Projected_2019020918_RR.gtif"

#################################################################################################################################
## The following functions are sub functions


def plot(df, gtif_file, output_folder, nb_hours = 1):# Plot a df in a png file

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})  

    # Plotting the heatmap
    im = ax.imshow(df, extent=[-9.965, 14.563084827903268, 39.4626295723437, 54.184031134174326], origin='upper', cmap='viridis',
                   vmin=0, vmax = 4*nb_hours
)


    # Plot the colorbar
    plt.colorbar(im, ax=ax, label=f"Precipitation during the past {nb_hours} hour(s) (mm)", pad = 0.1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

    name_file = gtif_file.split("/")[-1][:-5]
    plt.title(f"File : {name_file}")
    plt.savefig(f"{output_folder}/{name_file}.png")
    plt.close()


# This function sets to nan all the data that are not in France
def filter_data_by_france(df, shapefile_path, spatial_factor):

    # This loads french borders with a slight margin (depending on the SR factor)
    world = gpd.read_file(shapefile_path)
    france = world[world["NAME_FR"] == 'France']
    france = france.to_crs(epsg=2154)
    france_with_margin = france.geometry.buffer(spatial_factor*1000/3) 
    france_with_margin = france_with_margin.to_crs(epsg=4326)

    # Get the lat/lon of all our points
    lat_nw, lon_nw = 54.184031134174326, -9.965  # Nord West
    lat_se, lon_se = 39.4626295723437, 14.563084827903268  # Sud East
    n, m = df.shape 
    latitudes = np.linspace(lat_se, lat_nw, n)
    longitudes = np.linspace(lon_nw, lon_se, m) 
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes) # 2D version

    # We use a GeoPandas object to compute if the coords is in France or not
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]
    gdf = gpd.GeoDataFrame(geometry=points)
    gdf['in_france'] = gdf.geometry.within(france_with_margin.geometry.iloc[0])

    # We set to NaN the non French coordinates
    df_filtered = df.copy()
    for k in range(len(gdf)):
        if gdf.loc[k, "in_france"] == False:
            df_filtered.loc[gdf.loc[k, "geometry"].y, gdf.loc[k, "geometry"].x] = np.nan

    # The y-axis is upside down
    temp = df_filtered.to_numpy()
    flip = temp[::-1, :]
    df_final = pd.DataFrame(flip, index = df_filtered.index, columns = df_filtered.columns)

    return df_final

def good_format(df): # Set df to the good format 

    # We replace the 65535 by NaN
    df[df >= 65535] = np.nan

    # We replace the not french data by NaN (because it's fake measurements)
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    mask_lignes = (df.index >= 172) & (df.index <= 1134)
    mask_colonnes = (df.columns >= 375) & (df.columns <= 1725)

    df.loc[~mask_lignes, :] = np.nan 
    df.loc[:, ~mask_colonnes] = np.nan  


    # We convert the values to mm
    df = df / 10

    return df 


#################################################################################################################################
## The following functions are main functions


# This function plot a frame from the gtif file
# ONLY USE IT IF FOR HIGH RES
def plot_coméphore_high_res(gtif_file, output_folder, nb_hours = 1):

    with rasterio.open(gtif_file, 'r') as f:
        df = f.read(1)
        df = pd.DataFrame(df)
        df = good_format(df)
 
        plot(df, gtif_file, output_folder, nb_hours)

# This function plot a frame from the gtif file
# ONLY USE IT IF FOR CUSTOM RES (it's longer to compute)
# It only plot the data and filter it to the franch borders
def plot_coméphore_low_res(gtif_file, output_folder, spatial_factor = 30):

    with rasterio.open(gtif_file, 'r') as f:
        df = f.read(1)
        df = pd.DataFrame(df)

        fill_na_df = filter_data_by_france(df, "Blurring/filter_france", spatial_factor)

        plot(fill_na_df, gtif_file, output_folder)











