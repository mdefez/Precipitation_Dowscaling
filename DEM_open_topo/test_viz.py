import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.transform import xy

# Import config
from Coméphore.Config import working_directory, data_directory, topography_directory

# Charger le fichier .tif
with rasterio.open(topography_directory + "low_res.tif") as src:
    image = src.read(1)  # Lire la première bande
    width = src.width
    height = src.height
    print(width, height)
    print(image.shape, src.res)
    width = src.width
    height = src.height
    transform = src.transform

    # Coin haut gauche (0, 0)
    top_left = xy(transform, 0, 0, offset='ul')  # upper left

    # Coin bas droite (height-1, width-1)
    bottom_right = xy(transform, height-1, width-1, offset='lr')  # lower right

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})  

    # Plotting the heatmap
    im = ax.imshow(image, extent=[top_left[0], bottom_right[0], bottom_right[1], top_left[1]], origin='upper', cmap='terrain')

    # Plot the colorbar
    plt.colorbar(im, ax=ax, label=f"Elevation (mm)", pad = 0.1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

    plt.title("DEM")
    plt.savefig(topography_directory + "DEM_low_res.png")
    plt.close()

