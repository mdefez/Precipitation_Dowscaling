import sys
import pickle
import xarray as xr
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np
from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsRegressor
import h5py
import pyproj
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
import pikepdf
import fitz
import io 
from PIL import Image


# Main function, run the entire pipeline (editing figures, making predictions and computing metrics)

def main(com_file, ds):
    ref_fichier = com_file[10:-8]
    date = f" {com_file[16:18]} January 2019 {com_file[18:20]}H"

    chemin_image = os.path.join(os.getcwd(), "Simple_baseline_COMEPHORE/Images")
    fichier = os.path.join(chemin_image, ref_fichier)

    # Create a file for the tiemstamps if it does not exists
    if not os.path.exists(fichier):
        os.makedirs(fichier)  

    with PdfPages(f"Simple_baseline_COMEPHORE/Images/{ref_fichier}/figures.pdf") as pdf_fig:

        ########## Plotting ERA-5 ##########

        vmin = 0
        vmax = 4

        # Function to plot a map from a xarray
        def plot_map(df_plot, nom):
            # Create the figure with the geographical background
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})

            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')

            df_plot.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', 
                                cbar_kwargs={'label': "Precipitation during the passed hour (mm)"}, 
                                vmin=vmin, vmax=vmax)

            ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

            titre = nom.split("/")[-1][:-4] + "\n" + date
            ax.set_title(titre)

            # Save the fig in the pdf
            pdf_fig.savefig()
            plt.close()
            

        precip = ds.copy()

        # We map the 0-360 lon from ERA-5 to the -180/180 of com
        precip["longitude"] = xr.apply_ufunc(lambda x: x if x <= 180 else x - 360, precip["longitude"], vectorize = True)

        # Lat/lon of the lower_left / upper_right com_file
        lon_min_output = -9.965 
        lat_min_output = 39.4626295723437 
        lon_max_output = 14.563084827903268
        lat_max_output = 54.184031134174326

        # Filtering the xarray data to the concerned area
        # Be careful, ERA-5 goes to 0 to 360 in lon where coméphore is -180 to 180
        precip_fr = precip.where(
            (precip['latitude'] >= lat_min_output) & (precip['latitude'] <= lat_max_output) 
            & (precip['longitude'] <= lon_max_output) & (precip["longitude"] >= lon_min_output), drop=True
        )


        precip_fr = precip_fr.sortby("longitude")


        plot_map(precip_fr["tp"], f"Images/{ref_fichier}/Low resolution.png")


        ########## Plotting bicubic interpolation ###################################################################


        data_bicubic = precip_fr.copy()

        # Create a linspace with the expected lat/lon values corresponding to the high resolution
        latitudes = data_bicubic['latitude'].values
        longitudes = data_bicubic['longitude'].values
        precipitation = data_bicubic['tp'].values  

        new_latitudes = np.linspace(latitudes.min(), latitudes.max(), 1294)
        new_longitudes = np.linspace(longitudes.min(), longitudes.max(), 2156)

        new_lon, new_lat = np.meshgrid(new_longitudes, new_latitudes)
        lon, lat = np.meshgrid(longitudes, latitudes)

        # Naïve bicubic interpolation
        precipitation_fine = griddata(
            (lon.flatten(), lat.flatten()), 
            precipitation.flatten(), 
            (new_lon, new_lat), 
            method='cubic'
        )

        # Store & plot the data
        ds_interp = xr.DataArray(precipitation_fine, coords=[('latitude', new_latitudes), ('longitude', new_longitudes)])
        plot_map(ds_interp, f"Images/{ref_fichier}/Bicubic interpolation.png")

        # We invert the y axis because the (0, 0) value corresponds to the lower_left, we want it to match the com_file where
        # (0, 0) corresponds to the upper left
        ds_interp_iso = np.array(ds_interp)[::-1, :]

        ########## Plotting KNN ###################################################################

        data_bicubic = precip_fr.copy()

        # Create a linspace with the expected lat/lon values corresponding to the high resolution
        latitudes = data_bicubic['latitude'].values
        longitudes = data_bicubic['longitude'].values
        precipitation = data_bicubic['tp'].values  

        new_latitudes = np.linspace(latitudes.min(), latitudes.max(), 1294)
        new_longitudes = np.linspace(longitudes.min(), longitudes.max(), 2156)

        new_lon, new_lat = np.meshgrid(new_longitudes, new_latitudes)
        lon, lat = np.meshgrid(longitudes, latitudes)

        # Prepare the matrix format so that it can be put in KNN
        coords = np.vstack([lat.flatten(), lon.flatten()]).T
        new_coords = np.vstack([new_lat.flatten(), new_lon.flatten()]).T

        # Fit the kNN
        n_voisin = 5
        knn = KNeighborsRegressor(n_neighbors=n_voisin, weights="distance") 
        knn.fit(coords, precipitation.flatten())

        precipitation_fine = knn.predict(new_coords).reshape(new_latitudes.shape[0], new_longitudes.shape[0])

        # Store & plot the data
        ds_knn = xr.DataArray(precipitation_fine, coords=[('latitude', new_latitudes), ('longitude', new_longitudes)])
        plot_map(ds_knn, f"Images/{ref_fichier}/k-NearestNeighbors where k = {n_voisin}.png")

        # We invert the y axis because the (0, 0) value corresponds to the lower_left, we want it to match the com_file where
        # (0, 0) corresponds to the upper left
        ds_knn_iso = np.array(ds_knn)[::-1, :]

        ########## Plot the COM (ground truth) ###################################################################


        # Open the com file
        with rasterio.open("../../../downscaling/raw_data/Comephore/Projected_data/2019/COMEPHORE_2019_1/2019/" + com_file, 'r') as f:

            # Acquire the data
            df = f.read(1)
            df = pd.DataFrame(df)

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
                        vmin=vmin, vmax=vmax)


            # Plot the colorbar
            plt.colorbar(im, ax=ax, label="Precipitation during the passed hour (mm)")
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, edgecolor='black')

            ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

            ax.set_title("Ground Truth from Coméphore" + "\n" + date)

            pdf_fig.savefig()
            plt.close()

         
    ########## Metrics computing ###################################################################



    # This function computes all the metrics and store them in a generated pdf file
    def métrique(pred_ini, target, nom):
        with PdfPages(f"Simple_baseline_COMEPHORE/Images/{ref_fichier}/metrics {nom}.pdf") as pdf:

            target_array = np.asarray(target)

            ############ RMSE computing #####################################################################
            res = np.nanmean((pred_ini - target_array) ** 2) ** 0.5

            pdf_str = f"Root Mean Squared Error (RMSE): {str(res):.5} mm\n"


            ############ Plot the difference matrix (Prediction - Target) ####################################
            diff_matrix = (pred_ini - target_array) 

            # geographical background
            fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={"projection": ccrs.PlateCarree()})

            ax.set_extent([lon_min_output, lon_max_output, lat_min_output, lat_max_output])
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="dotted")


            img = ax.imshow(diff_matrix, extent=[lon_min_output, lon_max_output, lat_min_output, lat_max_output],
                            origin="upper", cmap="viridis", alpha=0.6) 

            # Colorbar plot
            plt.colorbar(img, orientation="vertical", label="Difference of precipitation in mm")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.title(f"Difference ({nom} - target) \n" + date)

            pdf.savefig()
            plt.close()


            ####### Quantile plot ###################################################################################

            # Flattening matrix into a 1D vector to compute some metrics
            pred = pred_ini.flatten()
            target_flat = target_array.flatten()

            # We plot "nb_quantiles" and not all the distribution given the high amount of samples
            nb_quantiles = 1000

            deciles1 = np.percentile(pred, np.linspace(1, 100, nb_quantiles))
            deciles2 = np.nanpercentile(target_flat, np.linspace(1, 100, nb_quantiles))  


            plt.figure(figsize=(8, 8))
            plt.scatter(deciles1, deciles2, color='b', label=f'Quantiles')

            plt.xlabel(f'{nom} quantiles')
            plt.ylabel('Target quantiles')
            plt.title(f'QQ plot {nom} VS Target')

            # Reference line at 45° 
            plt.plot([min(deciles1), max(deciles1)], [min(deciles1), max(deciles1)], linestyle='--', color='black', label='Reference line')

            plt.legend()

            plt.grid(True)
            pdf.savefig()
            plt.close()

            ######## Kolmogorov-Smirnov statistic ########################################################

            statistic, p_value = ks_2samp(pred[~np.isnan(pred)], target_flat[~np.isnan(target_flat)])
            pdf_str += f"\nKolmogorov-Smirnov Distance (KS): {str(statistic):.4} mm, p-value: {str(p_value):.4}\n"


            ######## Plot Target/Prediction histogram ####################################################
            plt.figure(figsize=(10, 8))

            plt.hist(pred, bins=200, density=True, label=f'{nom}', color='blue', histtype="step")
            plt.hist(target_flat, bins=200, density=True, label='Target', color='black', histtype="step")

            plt.xlabel('Precipitation')
            plt.ylabel('Density')
            plt.yscale("log")
            plt.title(f'Approached distribution {nom} VS Target')
            plt.legend()

            pdf.savefig()
            plt.close()


            ###### 99.999th Percentile Error computing #################################################

            p_true = np.nanpercentile(target_flat, 99.999) # 99.999th percentile of target probability dstribution
            p_pred = np.nanpercentile(pred, 99.999)  # 99.999th percentile of prediction probability dstribution
            error_99 = abs(p_true - p_pred)
            pdf_str += f"\n99.999th Percentile Error (PE) : {str(error_99):.4} mm\n"


            ####### Earth-Mover Distance (Wasserstein Distance) computing #############################
            emd = wasserstein_distance(pred[~np.isnan(pred)], target_flat[~np.isnan(target_flat)])

            pdf_str += f"\nEarth-Mover Distance (EMD) : {emd:.4}\n"


            ######## Spatial-Autocorrelation Error (SAE) computing ################################

            # Eventually write the code to compute the (right) SAE

            # Plotting the metrics in a white page
            plt.figure(figsize=(8, 8))  # Format A4 en pouces

            plt.text(0, 0.5,  pdf_str, fontsize=12, verticalalignment="center", family="monospace", horizontalalignment='left')

            plt.axis("off")
            pdf.savefig()
            plt.close()

    # Run the function for both the sallow models
    métrique(ds_interp_iso, df, "Bicubic Interpolation")
    métrique(ds_knn_iso, df, "kNN")



# This script is to be executed from another script with arguments as input

name = sys.argv[1]  # Name of the coméphore file
pickle_file = sys.argv[2]  # path to the ERA-5 file (actually only line corresponding to the expected timestamp)

# Extract the corresponding line
with open(pickle_file, 'rb') as f:
    ligne_extraite = pickle.load(f)

# Run the main function
main(name, ligne_extraite)