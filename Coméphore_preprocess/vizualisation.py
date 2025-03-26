# Use this script to plot (& save) an example of the projected Coméphore dataset
import numpy as np
import cartopy.crs as ccrs
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import pandas as pd
import cartopy.feature as cfeature

# open the GeoTIFF file

# Eventually change the strategy here ################################        ##############################################
file_path = "../../../downscaling/mdefez/Comephore/Projected_data/test/2154/2019/COMEPHORE_2019_2/2019/Projected_2019020718_RR.gtif"

with rasterio.open(file_path, "r") as src:
    # We read the image
    image = src.read(1)
    date = file_path.split("/")[-1][10:20]
    strategie_projection = file_path.split("/")[-5]

    # We process it to plot it correctly
    df = pd.DataFrame(image)

    # We replace the 65535 by NaN
    df = df.replace(65535, np.nan)

    # We replace the not french data by NaN (because it's fake measurements)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)

    mask_lignes = (df.index >= 172) & (df.index <= 1134)
    mask_colonnes = (df.columns >= 375) & (df.columns <= 1725)

    df.loc[~mask_lignes, :] = np.nan 
    df.loc[:, ~mask_colonnes] = np.nan  


    # We convert the values to mm
    df = df / 10

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})  

    # Plotting the heatmap
    im = ax.imshow(df, extent=[-9.965, 14.563084827903268, 39.4626295723437, 54.184031134174326], origin='upper', cmap='viridis',
                vmin=0, vmax=4)


    # Plot the colorbar
    plt.colorbar(im, ax=ax, label="Precipitation during the past hour (mm)", pad = 0.1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

    plt.title(date)
    plt.savefig(f"Coméphore_preprocess/Images/example_{strategie_projection}.png")
    plt.close()
