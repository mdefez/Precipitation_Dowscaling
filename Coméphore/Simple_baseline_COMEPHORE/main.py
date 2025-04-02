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

# Here you can tune the hyperparameter
temp_factor = 6
spatial_factor = 25

# Path of the ground truth
original_data_path = "../../../downscaling/mdefez/Comephore/Projected_data/2019/COMEPHORE_2019_2/2019"

# Where to store the temporal downscaled data
temporal_downscaled_data_path = "../../../downscaling/mdefez/Comephore/test_baseline/interm_data"

# Where to store the final processed data
input_data = "../../../downscaling/mdefez/Comephore/test_baseline/input_data"

# tool.plot_coméphore_low_res(input_data + '/aggregated_sample_Group 14.gtif', os.getcwd(), title = "test")


##############################################################################################################################
### First step : Process the data to create the inputs (blurring & downsampling)
##############################################################################################################################
data_already_processed = False 
if not data_already_processed:
    print("Downsampling data in space & time")
    tool.process_input(input_folder = original_data_path, 
                    interm_folder = temporal_downscaled_data_path,
                        output_folder = input_data, 
                        temp_factor = temp_factor, 
                        spatial_factor = spatial_factor)

##############################################################################################################################
### Second step : Super-resolving in time & space
##############################################################################################################################
# The predictions will be stored as array in a list (sorted by time)
print("Super resolving in time")
# Import the input into a (sorted) list of arrays
list_filename_input, list_input_low_temporal_res = fun.get_array_sorted_by_time(input_data)

################# Temporally super resolve ################################
temporal_method = {fun.temporal_interpolation} # Available time super resolving method

time_sr = fun.temporal_super_resolve(list_input_low_temporal_res = list_input_low_temporal_res,
                                     temp_factor = temp_factor,
                                     method = fun.temporal_interpolation)


# We now have a list of arrays with a 1 hour temporal resolution

################# Spatially super resolve ################################
print("Super resolving in space")
target_size = (1294, 2156) # high res of the Coméphore dataset

# Choose an avaiable space super resolving method between them
spatial_method = {fun.bicubic_interpolation, fun.nearest_neighbor}

nb_files_to_downsample = 10 # We compute some samples to visualize the results
list_output = [fun.bicubic_interpolation(arr, target_size) for arr in time_sr][:nb_files_to_downsample] 


##############################################################################################################################
### Third step : Computing the metrics and eventually plot some samples
##############################################################################################################################
print("Plotting examples")
# Get the target data in a array format & sort the list with respect to timesteps in the filename
list_filename_target, list_target = fun.get_array_sorted_by_time(original_data_path)


# Plot the computed examples
fun.plot_all_examples(nb_files=nb_files_to_downsample,
                      list_pred=list_output,
                      list_target=list_target,
                      list_filename=list_filename_target,
                      output_folder="Coméphore/Simple_baseline_COMEPHORE/plot_result",
                      spatial_factor=spatial_factor,
                        temp_factor=temp_factor)












