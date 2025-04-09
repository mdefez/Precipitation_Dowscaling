### The goal of this script is to target a domain (as left and north as possible) and slice it into 16 differents tiles
### For each tile, we will downsample (time & space) and save the data into a unique folder 

### The entire domain will be 400*400 km², thus each tile will be a square of 100*100km²

### In each subfolder, there will be a target dataset, with a 1*1km² high res, and an input dataset with a 25*25km² low res
### The subfolder will also contain the tile's DEM

import rasterio
from rasterio.transform import xy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys, os

sys.path.append(os.path.join(os.getcwd(), "Coméphore/Processing_input_data"))
import tools as tool

ex_file = "../../../downscaling/mdefez/Comephore/Projected_data/2020/COMEPHORE_2020_2/2020/Projected_2020021510_RR.gtif"

# Load a sample to compute locations
with rasterio.open(ex_file, 'r') as src:
    width = src.width
    height = src.height

    transform = src.transform

    df = src.read(1)
    df = pd.DataFrame(df)

    lon, lat = xy(transform, 420, 1190)
    # Print if needed
    # print(f"Coordinates : {lat}, {lon}")

# Coordinates of the rectangle of interest
upper_right_lat_lon = (49.40014444341452, -0.9717480552835571)
upper_right_pxl = (420, 790) # BE CAREFUL, the origin is upper left as it would be in a matrix

lower_left_lat_lon = (44.84947935798917, 3.578917030141797)
lower_left_pxl = (820, 1190)

# Define the coordinates of each tile
nb_tile_horizontal = 4
nb_tile_vertical = 4

hor_step = (lower_left_pxl[1] - upper_right_pxl[1]) / nb_tile_horizontal
vert_step = (lower_left_pxl[0] - upper_right_pxl[0]) / nb_tile_vertical

# Each tile is associated with its min/man pixel in row/column
dict_coord = {}

for hor in range(nb_tile_horizontal):
    for vert in range(nb_tile_vertical):
        key_name = f"horizontal_{hor}_vertical_{vert}"
        vertical_value = upper_right_pxl[0] + vert_step * vert
        hor_value = upper_right_pxl[1] + hor_step * hor 

        dict_coord[key_name] = ((int(vertical_value), int(vertical_value + vert_step)), 
                                (int(hor_value), int(hor_value + hor_step)))
    
   

# Return a tile to the corresponding timestep & position as an array 
# Timestep should be YYYYMMJJHH24
def get_a_tile(hor, vert, timestep):
    file = tool.get_path(str(timestep))

    with rasterio.open(file, 'r') as src:
        arr = np.array(src.read(1))
        idx_row, idx_column = dict_coord[f"horizontal_{hor}_vertical_{vert}"][0], dict_coord[f"horizontal_{hor}_vertical_{vert}"][1]

        tile = arr[idx_row[0]:idx_row[1], idx_column[0]:idx_column[1]]

    return tile

def get_dem(hor, vert):
    with rasterio.open("DEM_open_topo/low_res.tif", 'r') as src:
        arr = np.array(src.read(1))
        idx_row, idx_column = dict_coord[f"horizontal_{hor}_vertical_{vert}"][0], dict_coord[f"horizontal_{hor}_vertical_{vert}"][1]

        # We have to substract a value given that it is not the same origin
  
        dem = arr[idx_row[0] - upper_right_pxl[0]: idx_row[1] - upper_right_pxl[0], 
                  idx_column[0] - upper_right_pxl[1]: idx_column[1] - upper_right_pxl[1]]

    return dem

# Compute the timestep we need to have the target (to train or test)
def timestep_of_interest_target(year, n_days = 5):
    list_files = []
    for month in range(1, 13):
        month = str(month).zfill(2)

        for day in range(1, n_days + 1):
            day = str(day).zfill(2)

            for hour in range(24):
                if day == "01" and hour == 0:
                    continue
                hour = str(hour).zfill(2)

                file_to_add = f'{year}{month}{day}{hour}'
                list_files.append(file_to_add)

    return list_files

# Save the target for 2023 & 2024
def save_target_data(output_folder, year): # Usually output = "../../../downscaling/mdefez/Comephore/RNB/target_data"
    list_timestep = timestep_of_interest_target(year)

    for hor in range(nb_tile_horizontal):
        for vert in range(nb_tile_vertical):
            # Create a folder for each tile
            os.makedirs(os.path.join(output_folder, year, f"tile_hor_{hor}_vert_{vert}"), exist_ok=True)
            print(hor, vert)
            for timestep in list_timestep:
                tile = get_a_tile(hor, vert , timestep)

                np.save(f"{output_folder}/{year}/tile_hor_{hor}_vert_{vert}/{timestep}.npy", tile)

# save_target_data("../../../downscaling/mdefez/Comephore/RNB/target_data", "2024")

# Downsample the data to create the input
def save_input_data(output_folder, year): # Usually output = "../../../downscaling/mdefez/Comephore/RNB/input_data"
    for hor in range(nb_tile_horizontal):
        for vert in range(nb_tile_vertical):
            print(hor, vert)
            # Ge the dem file
            input_folder = f"../../../downscaling/mdefez/Comephore/RNB/target_data/{year}/tile_hor_{hor}_vert_{vert}"
            output_folder_tile = os.path.join(output_folder, year, f"tile_hor_{hor}_vert_{vert}")

            tool.process_input(input_folder = input_folder, 
                               interm_folder = f"../../../downscaling/mdefez/Comephore/RNB/interm_data/{year}/tile_hor_{hor}_vert_{vert}",
                                 output_folder = output_folder_tile, temp_factor = 6, spatial_factor = 25,
                                 area = "RNB")


# save_input_data("../../../downscaling/mdefez/Comephore/RNB/input_data", "2024")

# Save the DEM data tile by tile in the same folder as the coarse inpur data
def save_dem(output_folder): 
    os.makedirs(output_folder, exist_ok = True)
    for hor in range(nb_tile_horizontal):
        for vert in range(nb_tile_vertical):
            print(hor, vert)
            # Get to the ground truth we will "blur"
            dem = get_dem(hor, vert)

            np.save(f"{output_folder}/dem_hor_{hor}_vert_{vert}.npy", dem)

# save_dem("../../../downscaling/mdefez/Comephore/RNB/input_data/DEM")

def plot_example():
    fig, ax = plt.subplots(figsize=(6, 4))  

    input_data = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/2023/tile_hor_0_vert_0/beggining_2023010512_temp_factor_6_spatial_factor_25.npy"
    output_data = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data/2023/tile_hor_0_vert_0/2023010512.npy"
    # Plotting the heatmap
    im = ax.imshow(np.load(output_data), origin='upper', cmap='viridis')


    plt.colorbar(im, ax=ax, label=f"Precipitation during the past hour(s) (mm)", pad = 0.1)
    plt.title(output_data.split("/")[-1])
    plt.savefig("Coméphore/CV_pipeline/Images/target.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))  
    im = ax.imshow(np.load(input_data), origin='upper', cmap='viridis')
    print(np.load(input_data).shape, np.load(output_data).shape)

    plt.colorbar(im, ax=ax, label=f"Precipitation during the past hour(s) (mm)", pad = 0.1)
    plt.title(input_data.split("/")[-1])
    plt.savefig("Coméphore/CV_pipeline/Images/low_res.png")
    plt.close()

plot_example()

