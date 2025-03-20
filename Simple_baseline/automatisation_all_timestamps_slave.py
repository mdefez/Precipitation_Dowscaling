import sys
import pickle
import xarray as xr
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


# Main function, run the entire pipeline (editing figures, making predictions and computing metrics)

def main(cpc_file, ds):
    ref_fichier = cpc_file[:13]
    date = f" {cpc_file[6:8]} January 2019 {cpc_file[8:10]}H"

    chemin_image = os.path.join(os.getcwd(), "Simple_baseline/Images")
    fichier = os.path.join(chemin_image, ref_fichier)

    # Create a file for the tiemstamps if it does not exists
    if not os.path.exists(fichier):
        os.makedirs(fichier)  

    with PdfPages(f"Simple_baseline/Images/{ref_fichier}/figures.pdf") as pdf_fig:

        ########## Plotting ERA-5 ##########

        vmin = 0
        vmax = 4

        # Function to plot a map from a xarray
        def plot_map(df_plot, nom):
            # Create the figure with the geographical background
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

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
            pdf_fig.savefig(dpi = 100)
            plt.close()
            

        precip = ds.copy()

        # Lat/lon of the lower_left / upper_right CPC_file
        lon_min_output = 3.168779677002355 
        lat_min_output = 43.6290303456092 
        lon_max_output = 12.46232838782734 
        lat_max_output = 49.36326405028229

        # Filtering the xarray data to the concerned area
        precip_suisse = precip.where(
            (precip['latitude'] >= lat_min_output) & (precip['latitude'] <= lat_max_output) &
            (precip['longitude'] >= lon_min_output) & (precip['longitude'] <= lon_max_output), drop=True
        )

        plot_map(precip_suisse["tp"], f"Images/{ref_fichier}/Low resolution.png")


        ########## Plotting bicubic interpolation ###################################################################


        data_bicubic = precip_suisse.copy()

        # Create a linspace with the expected lat/lon values corresponding to the high resolution
        latitudes = data_bicubic['latitude'].values
        longitudes = data_bicubic['longitude'].values
        precipitation = data_bicubic['tp'].values  

        new_latitudes = np.linspace(latitudes.min(), latitudes.max(), 640)
        new_longitudes = np.linspace(longitudes.min(), longitudes.max(), 710)

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

        # We invert the y axis because the (0, 0) value corresponds to the lower_left, we want it to match the CPC_file where
        # (0, 0) corresponds to the upper left
        ds_interp_iso = np.array(ds_interp)[::-1, :]

        ########## Plotting KNN ###################################################################

        data_bicubic = precip_suisse.copy()

        # Create a linspace with the expected lat/lon values corresponding to the high resolution
        latitudes = data_bicubic['latitude'].values
        longitudes = data_bicubic['longitude'].values
        precipitation = data_bicubic['tp'].values  

        new_latitudes = np.linspace(latitudes.min(), latitudes.max(), 640)
        new_longitudes = np.linspace(longitudes.min(), longitudes.max(), 710)

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

        # We invert the y axis because the (0, 0) value corresponds to the lower_left, we want it to match the CPC_file where
        # (0, 0) corresponds to the upper left
        ds_knn_iso = np.array(ds_knn)[::-1, :]

        ########## Plot the CPC (ground truth) ###################################################################


        # Open the hdf5 file
        with h5py.File("../../../downscaling/MeteoSwiss_Products/CPCH/2019/" + cpc_file, 'r') as f:

            # Acquire the data
            df = f["/dataset1"]["data1"]["data"]
            df = pd.DataFrame(df[:])  

            # The data are coded through epsg_2056, we transform it into epsg_4326 to match the ERA-5 dataset
            epsg_2056 = pyproj.CRS("EPSG:2056")
            epsg_4326 = pyproj.CRS("EPSG:4326")

            transformer = pyproj.Transformer.from_crs(epsg_2056, epsg_4326, always_xy=True)

            # Lower_left and upper_right in epsg_2056
            lower_left_x, lower_left_y = 2255000, 840000  
            upper_right_x, upper_right_y = 2965000, 1480000 

            # Conversion into espg_4326
            lower_left_lon, lower_left_lat = transformer.transform(lower_left_x, lower_left_y)
            upper_right_lon, upper_right_lat = transformer.transform(upper_right_x, upper_right_y)


            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})  

            # Plotting the heatmap
            im = ax.imshow(df, extent=[lower_left_lon, upper_right_lon, lower_left_lat, upper_right_lat], origin='upper', cmap='viridis',
                        vmin=vmin, vmax=vmax)


            # Plot the colorbar
            plt.colorbar(im, ax=ax, label="Precipitation during the passed hour (mm)")
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, edgecolor='black')

            ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

            ax.set_title("Ground Truth from CPC" + "\n" + date)

            pdf_fig.savefig()
            plt.close()


         
    ########## Metrics computing ###################################################################



    # This function computes all the metrics and store them in a generated pdf file
    def métrique(pred_ini, target, nom):
        with PdfPages(f"Simple_baseline/Images/{ref_fichier}/metrics {nom}.pdf") as pdf:

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

name = sys.argv[1]  # Name of the cpc file
pickle_file = sys.argv[2]  # path to the ERA-5 file (actually only line corresponding to the expected timestamp)

# Extract the corresponding line
with open(pickle_file, 'rb') as f:
    ligne_extraite = pickle.load(f)

# Run the main funciton
main(name, ligne_extraite)