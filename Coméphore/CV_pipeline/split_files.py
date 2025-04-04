############################################################################################################################
### The goal of this file is to focus on a spatial domain (to avoid training our model on the whole dataset)
### Then we split our data into training and testing sets
### The splitting strategy is not random because it has to take into account the spatial and temporal autocorrelation of the data

### Concerning the spatial aspect : we split our domain into n regular parts


### Training set 
### We take the n th first days of each month of 2024

### Testing set 
### We take the n th first days of each month of 2023

### One have to keep in mind our training & testing datasets depend on the method we used image/video super resolution

input_data_path = "../../../downscaling/mdefez/Comephore/downsampled_data/input_data"
# format of the input file : 'beggining_YYYYMMDDHH_temp_factor_<temp_factor>_spatial_factor_<spatial_factor>.gtif'

def format_int(int): # Set int to the str format with length 2 (using 0 padding if needed)
    if int <= 9: 
        int = f"0{int}"
    else:
        int = str(int)

    return int

def split_files(temporal_factor, spatial_factor, year, n_days = 5):
    list_files = []
    for month in range(1, 13):
        month = format_int(month)

        for day in range(1, n_days + 1):
            day = format_int(day)

            for hour in range(1, 24 - temporal_factor + 2, temporal_factor):
                hour = format_int(hour)

                file_to_add = f'beggining_{year}{month}{day}{hour}_temp_factor_{temporal_factor}_spatial_factor_{spatial_factor}.gtif'
                list_files.append(file_to_add)

    return list_files

def list_training_files(temporal_factor, spatial_factor, n_days = 5, year = 2023):
    return split_files(temporal_factor, spatial_factor, year, n_days)

def list_testing_files(temporal_factor, spatial_factor, n_days = 5, year = 2024):
    return split_files(temporal_factor, spatial_factor, year, n_days)




















