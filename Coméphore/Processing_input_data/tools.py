# This file stores useful functions concerning the Coméphore dataset

from array import array
from datetime import time
from turtle import down
import pandas as pd
import numpy as np
import os
import warnings
import re
from typing import Union
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata
import geopandas as gpd
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
import ipdb

# Example of path to projected data
path_projected = f"../../../downscaling/mdefez/Comephore/Projected_data/test/9829/2019/COMEPHORE_2019_2/2019/Projected_2019020918_RR.gtif"

#################################################################################################################################
################################################## FIRST PART : PLOTTING ########################################################
#################################################################################################################################

# The goal of this part is to provide plotting & saving images

## The following functions are sub functions

# Plot the df, gtif_file is either the path or ""
def plot(df, gtif_file, output_folder, nb_hours = 1, title = None):# Plot a df in a png file
    os.makedirs(output_folder, exist_ok=True)
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

    # If gtif is the path
    if title == None:
        name_file = gtif_file.split("/")[-1][:-5]
    # If gtif is ""
    else:
        name_file = title

    plt.title(f"File : {name_file}")
    plt.savefig(f"{output_folder}/{name_file}.png")

    plt.close()

def save_pdf(pdf, df, gtif_file, nb_hours = 1, title = None): # Save the df plot in the specified pdf

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

    # If the gtif is a path
    if title == None:
        name_file = gtif_file.split("/")[-1][:-5]
        plt.title(f"File : {name_file}")

    # If the gtif is ""
    else:
        plt.title(f"{title}")
    pdf.savefig()
    plt.close()


def good_format(df, preprocess = False): # Set df to the good format 

    if preprocess: # Hide the fake values and transform values into mm
        df[df >= 65535] = np.nan
        df = df / 10

    # We replace the not french data by NaN (because it's fake measurements)
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    mask_lignes = (df.index >= 172) & (df.index <= 1134)
    mask_colonnes = (df.columns >= 375) & (df.columns <= 1725)

    df.loc[~mask_lignes, :] = np.nan 
    df.loc[:, ~mask_colonnes] = np.nan  

    return df 


#################################################################################################################################
## The following functions are main functions


# This function plot a frame from the gtif file
# ONLY USE IT IF FOR HIGH RES
# One can specify a path or a dataframe
def plot_coméphore_high_res(gtif_file : Union[str, pd.DataFrame], output_folder, nb_hours = 1, title = None, preprocess = False):
    if isinstance(gtif_file, str):
        with rasterio.open(gtif_file, 'r') as f:
            df = f.read(1)
            df = pd.DataFrame(df)
            
            df = good_format(df, preprocess)
            plot(df, gtif_file, output_folder, nb_hours, title)
    
    else:
        df = good_format(gtif_file, preprocess)
        plot(df, "", output_folder, nb_hours, title=title)



def save_pdf_coméphore_high_res(pdf, gtif_file, nb_hours = 1, title = None, preprocess = False):
    if isinstance(gtif_file, str):
        with rasterio.open(gtif_file, 'r') as f:
            df = f.read(1)
            df = pd.DataFrame(df)
            df = good_format(df)
    
            save_pdf(pdf, df, gtif_file, nb_hours, title)
    else:
        df = good_format(gtif_file, preprocess)
        save_pdf(pdf, df, "", nb_hours, title = title)

# This function plot a frame from the gtif file
# ONLY USE IT IF FOR CUSTOM RES (it's longer to compute)
# It only plot the data and filter it to the franch borders
def plot_coméphore_low_res(gtif_file : Union[str, pd.DataFrame], output_folder, nb_hours = 1, title = None):
    if isinstance(gtif_file, str):
        with rasterio.open(gtif_file, 'r') as f:
            df = f.read(1)
            df = pd.DataFrame(df)

            plot(df, gtif_file, output_folder, nb_hours, title)
    
    else:
        plot(gtif_file, "", output_folder, nb_hours, title = title)



def save_pdf_coméphore_low_res(pdf, gtif_file, nb_hours = 1, title = None):
    if isinstance(gtif_file, str):
        with rasterio.open(gtif_file, 'r') as f:
            df = f.read(1)
            df = pd.DataFrame(df)

            save_pdf(pdf, df, gtif_file, nb_hours, title)

    else:
        save_pdf(pdf, gtif_file, "", nb_hours, title = title)


#################################################################################################################################
########################################### SECOND PART : PROCESSING INPUT DATA #################################################
#################################################################################################################################

# The goal of this section is to provide an end-to-end function that process input samples, applying blurring and both temporal
# & spatial downsampling

# There are a few subfunctions for only one main function

# This functions takes as input a list of string containing a number and a second list of the same length, 
# It returns both lists ordered ascendingly with respect to the number in the first list 

def sort_string_list(list_1, list_2): 
    pattern = r'\d+(\.\d+)?'
    list_int = [int(re.search(pattern, text).group()) for text in list_1]

    df_sort = pd.DataFrame({"int" : list_int, "output" : list_2, "filename" : list_1})
    df_sort = df_sort.sort_values(by = "int", ascending=True)

    return list(df_sort["filename"]), list(df_sort["output"])

def gtif_to_array(gtif_file): # Convert a gtif file to an array
    with rasterio.open(gtif_file, 'r') as f:
        df = f.read(1)
        arr = pd.DataFrame(df).to_numpy() 

        return arr

def fill_na_arr(arr, margin = 6):
    x, y = np.indices(arr.shape)
    mask = ~np.isnan(arr)

    # For each nan, compute the distance to the closest non nan
    distances = distance_transform_edt(np.isnan(arr))

    arr_filled = arr.copy()

    # Keep the nan close enough 
    nan_x, nan_y = np.where(np.isnan(arr) & (distances <= margin))

    # Set the value to the corresponding closest value
    if nan_x.size > 0:
        x_valid = x[mask]
        y_valid = y[mask]
        values_valid = arr[mask]

        interpolated_values = griddata((x_valid, y_valid), values_valid, (nan_x, nan_y), method='nearest')
        arr_filled[nan_x, nan_y] = interpolated_values

    return arr_filled

def fill_na(df): # Fill the nan values to apply correctly the filters. We fill the nan by the closest (euclidian) non nan value
    
    arr = df.to_numpy()

    arr_filled = fill_na_arr(arr)

    return pd.DataFrame(arr_filled, index = df.index, columns = df.columns)


def apply_mean_filter(df, kernel_size):
    # We keep in memory the nan values before applying the kernel
    valid_mask = np.isnan(df)

    # We fill the nan 
    fill_df = fill_na(df)

    df_filtered = pd.DataFrame(uniform_filter(fill_df, size=kernel_size), columns=df.columns, index = df.index)
    # We set the value to NaN if it was initially NaN
    df_filtered[valid_mask] = np.nan

    return df_filtered

def downsampling(df, factor):
    arr = df.to_numpy()

    # We fill the nan 
    arr = fill_na_arr(arr)

    # Downsample
    new_shape = (arr.shape[0] // factor * factor, arr.shape[1] // factor * factor)

    # Tronquer l'array aux dimensions ajustées
    arr_downsampled = arr[:new_shape[0], :new_shape[1]]

    # Reshape et calcul de la moyenne
    arr_downsampled = arr_downsampled.reshape(new_shape[0] // factor, factor, new_shape[1] // factor, factor).mean(axis=(1, 3))
  
    # To compute the new coords
    lat_nw, lon_nw = 54.184031134174326, -9.965  # Coin nord-ouest (latitude, longitude)
    lat_se, lon_se = 39.4626295723437, 14.563084827903268  # Coin sud-est (latitude, longitude)

    new_lat = np.linspace(lat_se, lat_nw, arr_downsampled.shape[0])
    new_lon = np.linspace(lon_nw, lon_se, arr_downsampled.shape[1])

    # Convert back into df
    df_downsampled = pd.DataFrame(arr_downsampled, index = new_lat, columns = new_lon)

    return df_downsampled

# This function extracts a gpd object representing France
def get_france_geo_points(spatial_factor, path_shp = "Coméphore/Processing_input_data/filter_france"):

    # This loads french borders with a slight margin (depending on the SR factor)
    world = gpd.read_file(path_shp)
    france = world[world["NAME_FR"] == 'France']
    france = france.to_crs(epsg=2154)
    france_with_margin = france.geometry.buffer(spatial_factor*1000/3) 
    france_with_margin = france_with_margin.to_crs(epsg=4326)

    return france_with_margin

# This function stores the point coordinates we need to nan 
def nan_non_french_points(shape, france):
    # Get the lat/lon of all our points
    lat_nw, lon_nw = 54.184031134174326, -9.965  # Nord West
    lat_se, lon_se = 39.4626295723437, 14.563084827903268  # Sud East

    n, m = shape
    latitudes = np.linspace(lat_se, lat_nw, n)
    longitudes = np.linspace(lon_nw, lon_se, m) 
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes) # 2D version

    # We use a GeoPandas object to compute if the coords are in France or not
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]
    gdf = gpd.GeoDataFrame(geometry=points)
    gdf['in_france'] = gdf.geometry.within(france.geometry.iloc[0])

    # We store the non French coordinates
    to_nan = gdf.loc[gdf["in_france"] == False, "geometry"]

    return to_nan

# This function sets to nan the given coordinates
# The point of doing multiple functions is to not compute the coordinates to nan for each file (they don't change)
def set_to_nan(df, list_nan):
    n = df.shape[0]
    for nan in list_nan:
        lon = nan.x 
        lat = nan.y 

        lat_to_nan = n-1 - df.index.get_loc(lat)# For some reason the France map from GDO is upside down
        
        warnings.filterwarnings("ignore", category=FutureWarning)
        df.iloc[lat_to_nan][lon] = np.nan
        warnings.resetwarnings() 
        

    return pd.DataFrame(df, index=df.index, columns=df.columns)

# Blur and spatially downsample all the samples
def blur_and_spatial_downsampling(input_directory, output_directory, spatial_factor):
    france = get_france_geo_points(spatial_factor, path_shp="Coméphore/Processing_input_data/filter_france")
    os.makedirs(output_directory, exist_ok=True)

    # We sort the folder so that the function deals with file accordingly to the timestep it represents
    list_dir = os.listdir(input_directory)
    pattern = r'\d+(\.\d+)?'
    list_int = [int(re.search(pattern, text).group()) for text in list_dir]

    df_sort = pd.DataFrame({"int" : list_int, "filename" : list_dir})
    df_sort = df_sort.sort_values(by = "int", ascending=True)

    list_dir_sorted = list(df_sort["filename"])

    for filename in list_dir_sorted:
        output_name = f"{filename[:-5]}_spatial_factor_{spatial_factor}.gtif"
        print(f"Spatial processing file : {output_name}")
        with rasterio.open(os.path.join(input_directory, filename), 'r') as f:
            df = f.read(1)
            df = pd.DataFrame(df)
            # Downsampling with respect to the specified factor
            downsampled_df = downsampling(df, spatial_factor)

            if filename == list_dir_sorted[0]: # We compute it once
                low_res = downsampled_df.shape
                # Compute the points to set to nan given the new shape
                to_nan = nan_non_french_points(low_res, france)

            # Set to nan if not in france
            fill_na_df = set_to_nan(downsampled_df, to_nan)

            meta = f.meta # Save the meta to copy on downsampled file
            meta.update(dtype=rasterio.float32, count=1, driver='GTiff') 
            # Be careful, we must change the width & height given that we downsampled
            meta["width"] = fill_na_df.shape[1]
            meta["height"] = fill_na_df.shape[0]

            with rasterio.open(os.path.join(output_directory, output_name), 'w', **meta) as dst: # Save the downsampling file
                dst.write(fill_na_df.astype(rasterio.float32), 1)

# This function takes as input a folder where remain the samples we want to aggregate, the temporal factor to use
# and the output folder where we want to store the aggregated dataframes
def temporal_downsampling(input_directory, output_directory, temp_factor):

    os.makedirs(output_directory, exist_ok=True)

    # Dictionnary to store the groupped files
    time_groups = {}
    
    # Creating a df with the filename and the corresponding timestep

    liste_timestep = [int(filename[10:20]) for filename in os.listdir(input_directory)] # Format YYYYMMJJHH24
    liste_filename = os.listdir(input_directory)

    df_filename = pd.DataFrame({"filename" : liste_filename, "timestep" : liste_timestep})

    # Sort the df according to the timestep, luckily, increasing the timestep actually corresponds to move forward in time given the format file
    df_filename = df_filename.sort_values(by = "timestep", ascending=True)


    # Loop on the timesteps to add them in the right key
    # Given that they are sorted, we just have to iterate over the dataframe

    live_key = 0 
    count = 0   # Reset when we put enough frames into a key
    for k in range(len(df_filename)):
        # Load the file and save it to the right key
        with rasterio.open(os.path.join(input_directory, df_filename.iloc[k]["filename"])) as src:
            data = np.array(src.read(1))

            # Fill the fake value by nan
            data = np.where(data >= 65535, np.nan, data)

            # We divide by 10 to have mm
            data = data / 10

            # If the list does not exist yet
            if count == 0:
                timestep_name = df_filename.iloc[k]["timestep"]
                time_groups[f"beggining_{timestep_name}_temp_factor_{temp_factor}"] = []

            time_groups[f"beggining_{timestep_name}_temp_factor_{temp_factor}"].append(data)
        count += 1

        if count == temp_factor:
            live_key += 1
            count = 0


    # Average the files & save them
    for name in time_groups.keys():
        rasters = time_groups[name]

        summed_raster = np.mean(rasters, axis=0)
        
        # We put the date and hour range in the output filename
        output_filename = f"{name}.gtif"
        output_path = os.path.join(output_directory, output_filename)

        # We load any gtif to read and copy the metadata
        with rasterio.open(os.path.join(input_directory, df_filename.iloc[0]["filename"])) as src:
            meta = src.meta
            meta.update(dtype=rasterio.float32, count=1, driver='GTiff')  
            
            # Save the aggregated file
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(summed_raster.astype(rasterio.float32), 1)

            print(f"Temporal processing file : {output_filename}")



# The final function, that processes end-to-end the whole input dataset and save the output in a specified folder
def process_input(input_folder, interm_folder, output_folder, temp_factor, spatial_factor):

    temporal_downsampling(input_directory=input_folder,
                          output_directory=interm_folder,
                          temp_factor=temp_factor) # Temporal downsampling and saving the results in the intermediate folder

    blur_and_spatial_downsampling(input_directory= interm_folder,
                                  output_directory=output_folder,
                                  spatial_factor=spatial_factor) # blurring and spatial downsampling, saving the results in the output folder















