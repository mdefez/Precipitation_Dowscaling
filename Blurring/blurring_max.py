import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages

# Example of a file we want to blur
nom_com_file = f"../../../downscaling/mdefez/Comephore/Projected_data/test/9829/2019/COMEPHORE_2019_2/2019/Projected_2019020918_RR.gtif"

def plot(df, title, param, typ):# Plot a df in a pdf

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

    legend = ""
    if typ == "Gaussian filter":
        legend = "Sigma = " + str(param)
    if typ == "Mean filter":
        legend = "Kernel size = " + str(param)

    ax.set_title(title + "\n" + legend + "\n" + typ)

    pdf_fig.savefig()
    plt.close()

with PdfPages(f"Blurring/simple_blurring_example.pdf") as pdf_fig:
    with rasterio.open(nom_com_file, 'r') as f:
        df = f.read(1)
        df = pd.DataFrame(df)

        # Plot the real data
        plot(df, "Original data", None, "No filter")

        # We compute the total of precipitation (the mass)
        mass_before = float(df.sum().sum())

        ####################################### Gaussian filter
        # Plot the slightly filtered data
        sigma = 3  # Filter parameter, linked to the pixel's neighbors weights. The larger sigma the more we take into account the farther pixel
        df_filtered = pd.DataFrame(gaussian_filter(df.replace(65535, np.nan), sigma=sigma), columns=df.columns)

        plot(df_filtered, "Blurred data", sigma, "Gaussian filter")

        # Plot the significantly filtered data
        sigma = 20  # Filter parameter, linked to the pixel's neighbors weights. The larger sigma the more we take into account the farther pixel
        df_filtered = pd.DataFrame(gaussian_filter(df.replace(65535, np.nan), sigma=sigma), columns=df.columns)

        plot(df_filtered, "Very Blurred data", sigma, "Gaussian filter")

        ############################# Mean filter
        new_df = df.copy()
        new_df = new_df.replace(65535, np.nan)
        valid_mask = np.isnan(new_df)
        new_df[valid_mask] = 0


        # Plot the slightly filtered data
        kernel = 10  # Filter parameter, number of pixel to take into account
        df_filtered = pd.DataFrame(uniform_filter(new_df, size=kernel), columns=df.columns)
        df_filtered[valid_mask] = np.nan
        plot(df_filtered, "Blurred data", kernel, "Mean filter")

        # Plot the significantly filtered data
        kernel = 100  # Filter parameter, number of pixel to take into account
        df_filtered = pd.DataFrame(uniform_filter(new_df, size=kernel), columns=df.columns)
        df_filtered[valid_mask] = np.nan
        
        plot(df_filtered, "Blurred data", kernel, "Mean filter")
        