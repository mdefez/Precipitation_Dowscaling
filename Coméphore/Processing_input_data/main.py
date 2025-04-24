import slicing_area as slice 

spatial_factor = 1
temp_factor = 3

download_target = True # If we also want to download the target data

for year in [2023, 2024]:
    if download_target == True:
        slice.save_target_data("/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data", str(year))
    slice.save_input_data("/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data", str(year), spatial_factor, temp_factor)