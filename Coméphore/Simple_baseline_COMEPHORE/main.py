# This file execute the whole pipeline :
# 1) Processing the data 
# 2) Super-resolving in time & space
# 3) Computing the metrics and eventually plot some samples

import os 
import sys
import ipdb

sys.path.append(os.path.join(os.getcwd(), "Coméphore/Simple_baseline_COMEPHORE"))
import functions as fun

spatial_factor = 25
temp_factor = 6

# If we want to work on the little dataset (02/2019)
original_data_path = "../../../downscaling/mdefez/Comephore/Projected_data/2019/COMEPHORE_2019_2/2019"
input_data = "../../../downscaling/mdefez/Comephore/lil_dataset/input_data"


##############################################################################################################################
### First step : Super-resolving in time & space
##############################################################################################################################
# The predictions will be stored as array in a list (sorted by time)
print("Super resolving in time")
# Import the input into a (sorted) list of arrays
list_filename_input, list_input_low_temporal_res = fun.get_array_sorted_by_time(input_data)

################# Temporally super resolve ################################
available_temporal_method = [fun.temporal_interpolation] # Available time super resolving method
temporal_method = fun.temporal_interpolation

time_sr, list_filename_temporally_sr = fun.temporal_super_resolve(list_input_low_temporal_res = list_input_low_temporal_res,
                                     temp_factor = temp_factor,
                                     list_filename_input = list_filename_input,
                                     method = temporal_method) # Compute the temporally SR files and the filenames


# We now have a list of arrays with a 1 hour temporal resolution

################# Spatially super resolve ################################
print("Super resolving in space")
target_size = (1294, 2156) # high res of the Coméphore dataset

# Choose an avaiable space super resolving method between them
available_method = {fun.bicubic_interpolation, fun.nearest_neighbor}
spatial_method = fun.bicubic_interpolation

nb_files_to_downsample = 10 # We compute some samples to visualize the results

list_prediction, list_filename_prediction = fun.spatially_downsample(list_low_res = time_sr,
                      target_size = target_size,
                        method = spatial_method, 
                        nb_files_to_downsample = nb_files_to_downsample, 
                        list_filename_low_res = list_filename_temporally_sr)


##############################################################################################################################
### Third step : Computing the metrics and eventually plot some samples
##############################################################################################################################
print("Plotting examples")

# Get the target data in a array format & sort the list with respect to timesteps in the filename
list_filename_target, list_target = fun.get_array_sorted_by_time(original_data_path)
list_filename_target = fun.adapting_target_name(list_filename_target) # Making the filename more understandable

# Plot the computed examples
fun.plot_all_examples(nb_files = nb_files_to_downsample,
                      list_pred = list_prediction,
                      list_target = list_target,
                      list_filename_pred = list_filename_prediction,
                      list_filename_target = list_filename_target,
                      output_folder = "Coméphore/Simple_baseline_COMEPHORE/plot_result",
                      spatial_factor = spatial_factor,
                        temp_factor = temp_factor)












