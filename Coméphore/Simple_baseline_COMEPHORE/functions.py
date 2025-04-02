# This file aims to store useful functions for the simplebaseline pipeline

from ast import Call
from os import times
from turtle import title
import pandas as pd
import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
import sys 
import os
from typing import Callable

sys.path.append(os.path.join(os.getcwd(), "Coméphore/Processing_input_data"))
import tools as tool

################################################################################################################################
######################## PART 1 : Utility functions ###################################################################
################################################################################################################################

# This function takes a folder in argument (containing gtif files)
# It transforms the files to array returns a list of those arrays
# The list is sorted (ascendigly) by time, which means the first item is the array corresponding to the first timestep
def get_array_sorted_by_time(folder_gtif_file):
    list_filename = [gtif_file for gtif_file in os.listdir(folder_gtif_file)] # Get the list of the filenames
    list_array = [tool.gtif_to_array(os.path.join(folder_gtif_file, gtif_file)) for gtif_file in os.listdir(folder_gtif_file)] # Get the list of the corresponding array

    # Sort the list with respect to time
    list_filename, list_array = tool.sort_string_list(list_filename, list_array) # Sort both list according to the timestep in the filename

    return list_filename, list_array

################################################################################################################################
######################## PART 2 : SUPER RESOLUTION FUNCTIONS ###################################################################
################################################################################################################################

# Spatial bicubic interpolation
def bicubic_interpolation(arr, target_size): # target size in (row, column) 
    arr = tool.fill_na_arr(arr, margin = 3) # We fillna as a padding method (that's mirror padding basically)
    arr_augmented = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

    return arr_augmented
    
# Spatial nearest neighbor interpolation
def nearest_neighbor(arr, target_size): # Apply the one nearest neighbor
    return cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    



# Temporal interpolation
# This function takes 2 gtif files as input and a temporal factor
# It then returns a list of all the intermediate array, including the first one ( not hte second otherwise it will be represented 2 times)
def temporal_interpolation(array_1, array_2, temp_factor):
        return [(array_1 * (temp_factor - i) / temp_factor + array_2 * i / temp_factor) for i in range(temp_factor)]

# This function super resolves the folder into the specified temporal SR factor according to a method
def temporal_super_resolve(list_input_low_temporal_res, temp_factor, method : Callable):
    time_sr = [] # Stores the final SR frames

    for k in range(len(list_input_low_temporal_res)-1): # Compute every low res frame according to the method
        augmented_data = method(list_input_low_temporal_res[k], 
                                                    list_input_low_temporal_res[k+1], 
                                                    temp_factor = temp_factor)
        for arr in augmented_data:
            time_sr.append(arr)

    return time_sr



################################################################################################################################
################################### PART 2 : Plot & Metrics ###################################################################
################################################################################################################################



# Compute metrics for the file corresponding to the timestep and eventualy save them as pdf
# Takes array as input
def métrique(pred_ini : np.ndarray, target : np.ndarray, timestep):

    with PdfPages(f"Coméphore/Simple_baseline_COMEPHORE/Images/{timestep}/figures.pdf") as pdf:

        ############ RMSE computing #####################################################################
        res = np.nanmean((pred_ini - target) ** 2) ** 0.5

        pdf_str = f"Root Mean Squared Error (RMSE): {str(res):.5} mm\n"


        ####### Quantile plot ###################################################################################

        # Flattening matrix into a 1D vector to compute some metrics
        pred = pred_ini.flatten()
        target_flat = target.flatten()

        # We plot "nb_quantiles" and not all the distribution given the high amount of samples
        nb_quantiles = 1000

        deciles1 = np.percentile(pred, np.linspace(1, 100, nb_quantiles))
        deciles2 = np.nanpercentile(target_flat, np.linspace(1, 100, nb_quantiles))  


        plt.figure(figsize=(8, 8))
        plt.scatter(deciles1, deciles2, color='b', label=f'Quantiles')

        plt.xlabel(f'Prediction quantiles')
        plt.ylabel('Target quantiles')
        plt.title(f'QQ plot Prediction VS Target')

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

        plt.hist(pred, bins=200, density=True, label=f'Prediction', color='blue', histtype="step")
        plt.hist(target_flat, bins=200, density=True, label='Target', color='black', histtype="step")

        plt.xlabel('Precipitation')
        plt.ylabel('Density')
        plt.yscale("log")
        plt.title(f'Approached distribution Prediction VS Target')
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

        ############# Plotting the metrics in a white page ###################################
        plt.figure(figsize=(8, 8))  # Format A4 en pouces

        plt.text(0, 0.5,  pdf_str, fontsize=12, verticalalignment="center", family="monospace", horizontalalignment='left')

        plt.axis("off")
        pdf.savefig()
        plt.close()


# Plot the prediction vs the ground truth
def plot_pred_truth(pred : np.ndarray, target : np.ndarray, filename, output_folder, spatial_factor, temp_factor):

    timestep = filename[10:20]
    title_pred = f"Prediction {timestep}\nSpatial SR factor : {spatial_factor} km\nTemporal SR factor : {temp_factor} hours"
    title_target = f"Ground truth {timestep}"

    pred = pd.DataFrame(pred)
    tool.plot_coméphore_high_res(pred, output_folder + "/predictions", title = title_pred)

    target = pd.DataFrame(target)
    tool.plot_coméphore_high_res(target, output_folder + "/target", title = title_target, preprocess = True)

def plot_all_examples(nb_files, list_pred, list_target, list_filename, output_folder, spatial_factor, temp_factor):
    for sample in range(nb_files):
        plot_pred_truth(pred = list_pred[sample], 
                        target = list_target[sample], filename = list_filename[sample],
                        output_folder = output_folder,
                        spatial_factor=spatial_factor,
                        temp_factor=temp_factor)











