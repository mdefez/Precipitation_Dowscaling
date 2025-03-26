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



# Example of a file we want to process
nom_com_file = f"../../../downscaling/mdefez/Comephore/Projected_data/test/9829/2019/COMEPHORE_2019_2/2019/Projected_2019020918_RR.gtif"

def plot(df, typ, sr_factor):# Plot a df in a png file

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})  

    # Plotting the heatmap
    im = ax.imshow(df, extent=[-9.965, 14.563084827903268, 39.4626295723437, 54.184031134174326], origin='upper', cmap='viridis',
    vmin=0, vmax=4)


    # Plot the colorbar
    plt.colorbar(im, ax=ax, label="Precipitation during the past hour (mm)", pad = 0.1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

    plt.title(f"Timestamp : {nom_com_file.split("/")[-1][10:20]}\nBlurred data with {typ} filter, {dico_name_params[typ]} : {params[typ]} {dico_units[typ]} \nSpatial Downsampling by factor {sr_factor}")
    plt.savefig("Blurring/process_example.png")
    plt.close()


def good_format(df): # Set df to the good format 

    # We replace the 65535 by NaN
    df = df.replace(65535, np.nan)

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

def fill_na(df): # Fill the nan values to apply correctly the filters. We fill the nan by the closest (euclidian) non nan value
    arr = df.to_numpy()

    x, y = np.indices(arr.shape)
    mask = ~np.isnan(arr)

    x_valid = x[mask]
    y_valid = y[mask]
    values_valid = arr[mask]

    arr_filled = griddata((x_valid, y_valid), values_valid, (x, y), method='nearest')

    return pd.DataFrame(arr_filled, index = df.index, columns = df.columns)

def apply_gaussian_filter(df, sigma):
    df_filtered = pd.DataFrame(gaussian_filter(df, sigma=sigma), columns=df.columns)

    return df_filtered

def apply_mean_filter(df, kernel_size):
    # We keep in memory the nan values before applying the kernel
    valid_mask = np.isnan(df)

    # We fill the nan 
    fill_df = fill_na(df)

    df_filtered = pd.DataFrame(uniform_filter(fill_df, size=kernel_size), columns=df.columns, index = df.index)
    # We set the value to NaN if it was initially NaN
    df_filtered[valid_mask] = np.nan

    return df_filtered

def conserving_mass(df_input, df_output):
    # We compute the total of precipitation (the mass) before & after the transformation to make sure it's the same
    mass_before = float(df_input.sum().sum())
    mass_after = float(df_output.sum().sum())

    return df_output * mass_before / mass_after

def downsampling(df, factor):
    # We fill the nan 
    fill_df = fill_na(df)

    arr = fill_df.to_numpy()

    # Downsample
    arr_reshaped = arr[:arr.shape[0] // factor * factor, :arr.shape[1] // factor * factor].reshape(
        arr.shape[0] // factor, factor, arr.shape[1] // factor, factor
    )

    arr_downsampled = arr_reshaped.mean(axis=(1, 3))

    # To compute the new coords
    lat_nw, lon_nw = 54.184031134174326, -9.965  # Coin nord-ouest (latitude, longitude)
    lat_se, lon_se = 39.4626295723437, 14.563084827903268  # Coin sud-est (latitude, longitude)

    new_lat = np.linspace(lat_se, lat_nw, arr_downsampled.shape[0])
    new_lon = np.linspace(lon_nw, lon_se, arr_downsampled.shape[1])

    # Convert back into df
    df_downsampled = pd.DataFrame(arr_downsampled, index = new_lat, columns = new_lon)

    return df_downsampled

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

# Open the Coméphore file (projected in WSG 84)
with rasterio.open(nom_com_file, 'r') as f:
    df = f.read(1)
    df = pd.DataFrame(df)

    df = good_format(df)

    params = {"mean" : 10, "gaussian" : 1} # Respectively kernel_size & sigma
    dico_filter = {"mean" : apply_mean_filter, "gaussian" : apply_gaussian_filter}
    dico_name_params = {"mean" : "kernel size", "gaussian" : "sigma"}
    dico_units = {"mean" : "", "gaussian" : "mm"}

    # Choose here the type of filter to use
    filter_to_use = "gaussian"

    filtered_df = dico_filter[filter_to_use](df, params[filter_to_use])

    # Conservating the mass
    conserved_df = conserving_mass(df_input = df, df_output = filtered_df)

    # Downsampling with respect to the specified factor
    spatial_factor = 10
    downsampled_df = downsampling(conserved_df, spatial_factor)

    # Set to nan if not in france
    path_shp = "Blurring/filter_france"
    fill_na_df = filter_data_by_france(downsampled_df, path_shp, spatial_factor)

    plot(fill_na_df, typ = filter_to_use, sr_factor=spatial_factor)











