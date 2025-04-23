import slicing_area as slice 

spatial_factor = 1
temp_factor = 3

for year in [2023, 2024]:
    slice.save_input_data("/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data", year, spatial_factor, temp_factor)