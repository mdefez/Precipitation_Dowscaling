import slicing_area as slice 

# Import config
from Coméphore.Config import working_directory, data_directory

spatial_factor = 10
temp_factor = 3

download_target = False # If we also want to download the target data

for year in [2023, 2024]:
    if download_target == True:
        slice.save_target_data(data_directory + "target_data", str(year))
    slice.save_input_data(data_directory + "input_data", str(year), spatial_factor, temp_factor)