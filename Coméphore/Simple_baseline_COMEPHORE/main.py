# This file execute the whole pipeline :
# 1) Processing the data 
# 2) Super-resolving in time & space
# 3) Computing the metrics and eventually plot some samples

import os 
import sys
import ipdb

sys.path.append(os.path.join(os.getcwd(), "Coméphore/Processing_input_data"))
sys.path.append(os.path.join(os.getcwd(), "Coméphore/Simple_baseline_COMEPHORE"))
import tools as tool
import functions as fun
import rasterio
import pandas as pd

# Here you can tune the hyperparameter
temp_factor = 6
spatial_factor = 25

# Path of the ground truth
original_data_path = "../../../downscaling/mdefez/Comephore/Projected_data/2019/COMEPHORE_2019_2/2019"

# Where to store the temporal downscaled data
temporal_downscaled_data_path = "../../../downscaling/mdefez/Comephore/test_baseline/interm_data"

# Where to store the final processed data
input_data = "../../../downscaling/mdefez/Comephore/test_baseline/input_data"

##############################################################################################################################
### First step : Process the data to create the inputs (blurring & downsampling)
##############################################################################################################################
data_already_processed = False 
if not data_already_processed:
    tool.process_input(input_folder = original_data_path, 
                    interm_folder = temporal_downscaled_data_path,
                        output_folder = input_data, 
                        temp_factor = temp_factor, 
                        spatial_factor = spatial_factor)

##############################################################################################################################
### Second step : Super-resolving in time & space
##############################################################################################################################
# The predictions will be stored as array in a list (sorted by time)

# Save the input to array 
list_filename = [os.path.join(input_data, gtif_file) for gtif_file in os.listdir(input_data)]
list_input_low_temporal_res = [tool.gtif_to_array(os.path.join(input_data, gtif_file)) for gtif_file in os.listdir(input_data)]

# Sort the list with respect to time
list_filename, list_input_low_temporal_res = tool.sort_string_list(list_filename, list_input_low_temporal_res)

# Temporally super resolve
time_sr = []
for k in range(len(list_input_low_temporal_res)-1):
    augmented_data = fun.temporal_interpolation(list_input_low_temporal_res[k], 
                                                list_input_low_temporal_res[k+1], 
                                                temp_factor = temp_factor)
    for arr in augmented_data:
        time_sr.append(arr)


# We now have a list of arrays with a 1 hour temporal resolution
# Spatially super resolve 
target_size = (1294, 2156) # high res of the Coméphore dataset

# Choose one out of two method
# method = fun.nearest_neighbor
method = fun.bicubic_interpolation

list_output = [method(arr, target_size) for arr in time_sr]


##############################################################################################################################
### Third step : Computing the metrics and eventually plot some samples
##############################################################################################################################

# Get the target data 
list_filename_target = [os.path.join(original_data_path, gtif_file).split("/")[-1] for gtif_file in os.listdir(original_data_path)]
list_target = [tool.gtif_to_array(os.path.join(original_data_path, gtif_file)) for gtif_file in os.listdir(original_data_path)]

# Sort the list with respect to time
list_filename_target, list_target = tool.sort_string_list(list_filename_target, list_target)


# Plot an example
for sample in range(7):
    fun.plot_pred_truth(pred = list_output[sample], 
                        target = list_target[sample], filename = list_filename_target[sample],
                        output_folder = "Coméphore/Simple_baseline_COMEPHORE/test_result")













