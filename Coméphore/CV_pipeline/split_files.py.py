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

input_data_path = "../../../downscaling/mdefez/Comephore/test_baseline/input_data"
# format of the input file : 'aggregated_sample_Group {int}.gtif' where int is the number of generated file since the beggining

n_days = 5 # The number of first days to consider for each month

def relevant_aggregated_files(n, temporal_factor):
    files_per_month = (24 / temporal_factor) * n # Number of consecutive aggregated files to take per month

    





















