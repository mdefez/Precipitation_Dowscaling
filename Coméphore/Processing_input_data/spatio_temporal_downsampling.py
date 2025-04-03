import tools as tool

# Here one can tune both SR factor
temp_factor = 6
spatial_factor = 25

# If we want to downsample a specific folder
download_specific_folder = False

if download_specific_folder == True:
    original_data_path = "../../../downscaling/mdefez/Comephore/Projected_data/2019/COMEPHORE_2019_2/2019"

    # Where to store the temporal downscaled data, it gets deleted at the end
    temporal_downscaled_data_path = "../../../downscaling/mdefez/Comephore/lil_dataset/interm_data"

    # Where to store the final processed data
    input_data = "../../../downscaling/mdefez/Comephore/lil_dataset/input_data"
    print(f"Downsampling lil dataset month/year : 2/2019")
    tool.process_input(input_folder = original_data_path, 
                        interm_folder = temporal_downscaled_data_path,
                            output_folder = input_data, 
                            temp_factor = temp_factor, 
                            spatial_factor = spatial_factor)
    

else: # Otherwise, we downsample all the data from the same year
    year = 2024

    temporal_downscaled_data_path = "../../../downscaling/mdefez/Comephore/downsampled_data/interm_data"
    input_data = "../../../downscaling/mdefez/Comephore/downsampled_data/input_data"

    for month in range(1, 13):
        original_data_path = f"../../../downscaling/mdefez/Comephore/Projected_data/{year}/COMEPHORE_{year}_{month}/{year}"
        print(f"Downsampling month/year : {month}/{year}")
        tool.process_input(input_folder = original_data_path, 
                        interm_folder = temporal_downscaled_data_path,
                            output_folder = input_data, 
                            temp_factor = temp_factor, 
                            spatial_factor = spatial_factor)



















