# This script aims to provide useful functions concerning the normalization process and saving process


from torch.utils.data import DataLoader, Dataset
import torch
import os, pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get stats from the training set for normalizing
def get_stats(flatten_data, strategy):
    if strategy == "Standard":
        mean = flatten_data.mean()
        std = flatten_data.std()
        return mean, std 
    
    elif strategy == "min_max":
        min = flatten_data.min()
        max = flatten_data.max()
        return min, max
    
    elif strategy == "Robust":
        median = flatten_data.median()
        q1 = flatten_data.quantile(0.25)
        q3 = flatten_data.quantile(0.75)
        iqr = q3 - q1
        return median, iqr

# Compute the stats for the whole dataset
def compute_stats(train_dataset, strat_precip, strat_channel):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    all_data_precip = []
    all_data_channel = []

    for precip, channel, target, time_idx in train_loader:
        # precip: List of (B, 1, 1, H, W)

        for frame in precip: # Add each batch of the list to the total amount of data
            all_data_precip.append(frame)

        all_data_channel.append(channel) # Channel is already a tensor

    # Concat everything shape: (N, H, W)
    all_precip = torch.cat(all_data_precip, dim=0).squeeze() # Shape (N, 1, 1, H, W)
    all_channel = torch.cat(all_data_channel, dim=0)              # Shape (N', H, W)

    # flatten 
    all_precip_flat = all_precip.flatten()
    all_channel_flat = all_channel.flatten()

    # Get the corresponding stats
    stats_precip = get_stats(all_precip_flat, strat_precip)
    stats_channel = get_stats(all_channel_flat, strat_channel)

    return stats_precip, stats_channel

# Returns the function to apply at each sample from the training & testing set
def compute_transformation(train_dataset, strat_precip, strat_channel): 

    list_strat = [strat_precip, strat_channel]
    list_stats = compute_stats(train_dataset, strat_precip, strat_channel)
    list_transfo = []

    for k in range(len(list_strat)):
        strat = list_strat[k]
        if strat == "Standard":
            mean, std = list_stats[k]
            transformation = lambda x: (x - mean) / std

        elif strat == "min_max":
            min, max = list_stats[k]
            transformation = lambda x: (x - min) / (max - min)

        elif strat == "Robust":
            median, iqr = list_stats[k]
            transformation = lambda x: (x - median) / iqr     
        
        list_transfo.append(transformation)

    return list_transfo, list_stats

# Useful to save the transformation function
def get_transfo(strat, stats): 
    if strat == "Standard":
        mean, std = stats
        transformation = lambda x: (x - mean) / std

    elif strat == "min_max":
        min, max = stats
        transformation = lambda x: (x - min) / (max - min)

    elif strat == "Robust":
        median, iqr = stats
        transformation = lambda x: (x - median) / iqr 

    return transformation

# Takes the datasets as input and returns the normalized datasets(dem, inputs & targets)
class TransformedDataset(Dataset): 
    def __init__(self, base_dataset, transform_precip = None, transform_dem = None):
        self.base_dataset = base_dataset
        self.transform_precip = transform_precip
        self.transform_dem = transform_dem

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        low_res_tensors, channel, targets, time_idx = self.base_dataset[index]

        # Normalize the DEM
        if self.transform_dem != None:
            channel = self.transform_dem(channel)

        # Normalize the precip
        if self.transform_precip != None:
            for k in range(len(low_res_tensors)):
                low_res_tensors[k] = self.transform_precip(low_res_tensors[k])
            
            targets = self.transform_precip(targets)

        # Pass everything to the device
        channel, targets = channel.to(device), targets.to(device)
        for k in range(len(low_res_tensors)):
                low_res_tensors[k] = low_res_tensors[k].to(device)

        return low_res_tensors, channel, targets, time_idx



# Function that saves the deterministic model's weights in the file path
def save_model_deter(weights, name_of_the_run, spatial_factor, temp_factor):
    filepath = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/weights/spatial_{spatial_factor}_temp_{temp_factor}/'
    os.makedirs(filepath, exist_ok=True)
    torch.save({'model_state_dict': weights}, filepath + name_of_the_run + ".pth")

# Function that saves the diffusion model's weights in the file path
def save_model_diffu(weights, name_of_the_run, spatial_factor, temp_factor):
    filepath = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion/weights/spatial_{spatial_factor}_temp_{temp_factor}/'
    os.makedirs(filepath, exist_ok=True)
    torch.save({'model_state_dict': weights}, filepath + name_of_the_run + ".pth")

# Save some useful variables for computing the normalization on the test set
def save_transfo(output_path, best_stats_precip, best_stats_dem, strat_precip, strat_dem): 
    os.makedirs(output_path, exist_ok=True)

    list_precip = [stat.item() for stat in best_stats_precip]
    list_dem = [stat.item() for stat in best_stats_dem]

    with open(output_path + f"/best_stats_precip_{strat_precip}.pkl", "wb") as f:
        pickle.dump(list_precip, f)
    with open(output_path + f"/best_stats_dem_{strat_dem}.pkl", "wb") as f:
        pickle.dump(list_dem, f)

# Load the transformation function
def load_best_transform(file, strat_precip, strat_dem):
    with open(file + f"/best_stats_precip_{strat_precip}.pkl", "rb") as f:
        stats_precip = pickle.load(f)
    with open(file + f"/best_stats_dem_{strat_dem}.pkl", "rb") as f:
        stats_dem = pickle.load(f)

    best_transform_precip = get_transfo(strat=strat_precip, stats=stats_precip)
    best_transform_dem = get_transfo(strat=strat_dem, stats=stats_dem)
    best_transform = (best_transform_precip, best_transform_dem)

    return best_transform











