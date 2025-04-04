# The goal of this file is to build a big array corresponding to the whole (available) Coméphore area DEM data
# It uses as input all the DEM datasets corresponding to a 1° * 1° tile

# We have to build a matrix (list of list) where we store the array accordingly to their position in the area
# Then we can concatenate the matrix on the rows and on the columns

import os
import numpy as np
import rasterio
from skimage.measure import block_reduce
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

range_north = range(53, 38, -1) # We look at arrays until 54° (Coméphore is 54.2°)
range_west = range(170, 180)
range_east = range(0, 15)

def get_file(north, lon, east_or_west):
    filename = f"N{north}_00_{east_or_west}{str(lon).zfill(3)}"
    foldername = f"../../../downscaling/mdefez/DEM/data/{filename}/"

    try:
        file = foldername + os.listdir(foldername)[0] + f"/Copernicus_DSM_10_{filename}_00/DEM//Copernicus_DSM_10_{filename}_00_DEM.dt2"
        return file
    
    except:
        return None

facteur_de_réduction = 10
new_size = int(3601 / facteur_de_réduction) + 1

def get_array(file_path): # Get a (3601, 3601) array corresponding to the tile then meen pool it to 121*121
    if file_path == None: # If there is no data, we set to nan
        return np.full((new_size, new_size), np.nan)
    
    else:
        with rasterio.open(file_path) as src: # Add the tile's array to the row
            arr = src.read(1)

            if arr.shape != (3601, 3601): # If the array hasn't the good shape (N >= 50°), upsample it to the right one
                # It is 3601*1801, we just have to interpolate one column out of 2 linearly
                new_array = np.zeros((3601, 3601))

                new_array[:, ::2] = arr

                new_array[:, 1::2] = (arr[:, :-1] + arr[:, 1:]) / 2  
                
                mean_pooled_array = block_reduce(new_array, block_size=(facteur_de_réduction, facteur_de_réduction), func=np.mean)

            else:
                mean_pooled_array = block_reduce(arr, block_size=(facteur_de_réduction, facteur_de_réduction), func=np.mean)
        
        return mean_pooled_array



list_array = []

for north in range_north:
    row = []
    print(north)
    for west in range_west:
        file_path = get_file(north=north, lon=west, east_or_west="W")

        row.append(get_array(file_path))
    
    for east in range_east:
        file_path = get_file(north=north, lon=east, east_or_west="E")

        row.append(get_array(file_path))

    list_array.append(row)

# Transform the list into a big array
rows = [np.hstack(row) for row in list_array]
final_array = np.vstack(rows)

# Fill the nan with 0
final_array = np.nan_to_num(final_array, 0)

# Scale the array to the Coméphore resolution
scale_x = 1294 / final_array.shape[0]
scale_y = 2156 / final_array.shape[1]

array_resized = zoom(final_array, (scale_x, scale_y), order=1)

# Save and plot
np.save("DEM/DEM_Coméphore.npy", array_resized)

plt.imshow(array_resized, cmap='terrain')  
plt.colorbar(label='Height (m)')  
plt.title("DEM")
plt.savefig("DEM/DEM_coméphore_area.png")




















