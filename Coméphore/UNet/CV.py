# The goal of this script is to perform the block Cross Validation over the spatial domain
# This runs a k-fold CV and saves the best performing weights

from dataset import RainSuperResDataset
from train import train 
import torch 
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR, CyclicLR
from functools import partial
import torch.nn as nn
from loss import CustomLossTrain, LossTest 
import tools as tool 
from test import test 
import time
import pandas as pd

start_time = time.time()

# Check for GPU
print("CUDA available :", torch.cuda.is_available())
print("Number of GPUs :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name :", torch.cuda.get_device_name(0))

####################################################################################################################################################################################
############################################### CHANGE THE TRAINING FEATURES HERE ##############################################################################################
####################################################################################################################################################################################


# Super resolution factors
temp_factor = 1
spatial_factor = 10
n_inputs = 5        # Frames to take into account as input, the last one is the image to downscale

n_days_train = 8 # Only first n_days are used for each month
n_days_test = 5

# Choose wether we want to train/test/both
training = True 
normalizing = False # If False, the code will import the last normalizer saved
testing = True 

# Data directories
input_dir = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/spatial_{spatial_factor}_temp_{temp_factor}'          # Low res frames
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'        # High res targets
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'    # DEM

# Available models. One should choose a model and fill the corresponding parameters
available_model = ["UNet_with_attention"]
required_model_parameters = {"UNet_with_attention" : ("hard_constraint_mass", "n_inputs", "attention strategy")}

# Hard constraint mass
available_strategy_mass = [None, "additive", ("multiplicative", "a function type that operates on tensors")] # The function should apply element wise for tensors
def f_mass(x): # Function to apply element by element to the tensor. Be careful, it should not be zero when x = 0 and i thould not diverge when x is big
    return torch.exp(x)
treshold_constraint = 8 # Epoch where we should begin to apply conservative transformation

# Attention parameters
list_strat_attention = ["bottleneck", "encoder", None]
strat_attention = "encoder"

# Choice of the model and parameters
model = "UNet_with_attention"
model_parameters = (("multiplicative", f_mass), n_inputs, strat_attention)

# Training features
batch_size = 128
epochs = 20
learning_rate = 5e-5

# Loss function (used for training)
base_loss = nn.L1Loss       # Fill with nn.L1Loss or nn.MSELoss
conservative = False        # If we add the conservative mass soft constraint
lambda_conservative = 0.3
autocorrel = False          # If we add the autocorrelation soft constraint
lambda_autocorr = 0.3

loss_function = CustomLossTrain(base_loss=base_loss(),
                           lambda_conservative=lambda_conservative,
                           conservative=conservative,                  
                           covariance=autocorrel, 
                           lambda_covariance=lambda_autocorr)

name_loss = {nn.MSELoss : "l2", nn.L1Loss : "l1"}

# Metric (used for testing)
metric_test = [nn.L1Loss, nn.MSELoss]       # Fill by the metric to test the model on. The first one will be the main metric (used to compute the best/worst examples)
name_metric = [name_loss[metric] for metric in metric_test]     # Name of the metrics
df_metric = pd.DataFrame({"Name" : name_metric, "Metric" : metric_test})

metric = LossTest(df_metric=df_metric)

# All available scheduler strategies
dict_scheduler = {"Step decay" : ("Step decay", "epoch", partial(StepLR, step_size= (epochs // 3) + 1, gamma=0.1)), # Every step_size epoch, multiply the learning rate by gamma
                  "Cyclical" : ("Cyclical", "batch", partial(CyclicLR, base_lr=learning_rate, max_lr=learning_rate * 10, step_size_up=50, mode='triangular2'))} # The lr goes from min to max in a step_size period. After each cycle, the max value is divided by 2

scheduler = dict_scheduler["Cyclical"] # Choose the strategy here

# Normalization strategies
dict_strategies = ["Standard", "min_max", "Robust"]

strat_precip = "min_max"
strat_dem = "min_max"


# Design the name of the run
name_of_the_run = f"{name_loss[base_loss]}_input_{n_inputs}_n_days_{n_days_train}_attention_{strat_attention}_delay_constraint_{treshold_constraint}"

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# Check for valid settings
assert not (strat_attention != None and n_inputs == 1), "You want to compute attention with a 1-input sequence"
assert not (strat_attention == None and n_inputs != 1), "You want multiple inpus without computing attention"
assert epochs >= treshold_constraint, "The treshold for the mass conservation constraint is set after the number of epochs"
assert model in available_model, "Model not part of available model"
assert strat_attention in list_strat_attention, "Attention strategy not in the list of available attention strategies"
assert len(model_parameters) == len(required_model_parameters[model]), "Wrong number of model parameters filled"

# Loading all the datasets
dico_dataset = {}
for hor in range(4):
    for vert in range(4):
        dico_dataset[f"tile_hor_{hor}_vert_{vert}"] = RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, train=True, n_days=n_days_train, n_inputs=n_inputs, 
                                                                          temp_factor=temp_factor, spatial_factor=spatial_factor)


# This represents the 4 sub domains, we then perform a 4-fold CV on them
# We split the domain into "diagonals" so that the validation dataset 
splits = [
    [(0,0), (1, 1), (2, 2), (3, 3)],
    [(0,1), (1, 2), (2, 3), (3, 0)],
    [(0,2), (1, 3), (2, 0), (3, 1)],
    [(0,3), (1, 0), (2, 1), (3, 2)]
]

# Variables to find the best model
best_val_score = float('inf')
best_model_state = None

# CV pipeline 
if training:
    for k in range(len(splits)):
        print(f"Training of the split : {k+1}")
        # Compute the training / validating index
        val_idx = splits[k]
        train_idx_list = splits[:k] + splits[k+1:] 
        train_idx = []
        for x in train_idx_list:
            train_idx = train_idx + x

        list_train_dataset = [dico_dataset[f"tile_hor_{hor}_vert_{vert}"] for (hor, vert) in train_idx]
        list_val_dataset = [dico_dataset[f"tile_hor_{hor}_vert_{vert}"] for (hor, vert) in val_idx]

        print(f"Training tiles : {train_idx}")
        print(f"Validating tiles : {val_idx}")

        # Create the training & testing loaders for the split
        train_dataset = ConcatDataset(list_train_dataset)
        val_dataset = ConcatDataset(list_val_dataset)

        # Normalize the data according to the specified strategies

        if normalizing: # Compute the normalizer
            (transform_precip, transform_dem), (stats_precip, stats_dem) = tool.compute_transformation(train_dataset=train_dataset, strat_precip = strat_precip, strat_channel = strat_dem)
        else: # Load the normalizer
            best_transform = tool.load_best_transform(file = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/normalization",
                                         strat_dem=strat_dem, strat_precip=strat_precip)
            transform_precip, transform_dem = best_transform

        normalized_train_dataset = tool.TransformedDataset(base_dataset = train_dataset,
                                                        transform_precip = transform_precip,
                                                        transform_dem = transform_dem)
        normalized_val_dataset = tool.TransformedDataset(base_dataset = val_dataset,
                                                        transform_precip = transform_precip,
                                                        transform_dem = transform_dem)
        print("Data normalized")
        weights, loss = train(train_dataset=normalized_train_dataset, 
                            test_dataset=normalized_val_dataset, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            strategy_scheduler=scheduler, 
                            learning_rate=learning_rate, 
                            loss_function=loss_function,
                            split = k,
                            name_run = name_of_the_run,
                            temp_factor = temp_factor,
                            spatial_factor = spatial_factor,
                            asked_model=model,
                            model_parameters=model_parameters,
                            treshold_constraint=treshold_constraint)

        print(f"Loss on split {k+1}: {loss}")
        tool.save_model(weights, name_of_the_run)     # Save model at each split (so that we eventually don't have to wait the end of the CV). To be deleted.
        if normalizing:
            tool.save_transfo(output_path = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/normalization", 
                          best_stats_precip = stats_precip, best_stats_dem = stats_dem, strat_dem = strat_dem, strat_precip = strat_precip) # Same for the normalizer

        # Keep in memory the best model
        if loss < best_val_score:
            best_val_score = loss
            best_model_state = weights
            if normalizing:
                best_stats_precip, best_stats_dem = stats_precip, stats_dem

    # Save the best model & the associated normalization
    tool.save_model(weights, name_of_the_run)
    if normalizing:
        tool.save_transfo(output_path = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/normalization", 
                 best_stats_precip = best_stats_precip, best_stats_dem = best_stats_dem, strat_dem = strat_dem, strat_precip = strat_precip)
    print(f"\nBest model saved with score: {best_val_score}")

end_time_training = time.time()
print(f"Training : {int((end_time_training - start_time) / 60)} minutes")

# Model testing and vizualisation of some plots
if testing:
    best_transform = tool.load_best_transform(file = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/normalization",
                                         strat_dem=strat_dem, strat_precip=strat_precip)

    test(input_dir = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/spatial_{spatial_factor}_temp_{temp_factor}',
        output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data',
        channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM',
        spatial_factor=spatial_factor, temp_factor=temp_factor, n_inputs=n_inputs, name_of_the_run=name_of_the_run,
        best_transform=best_transform, criterion=metric, batch_size=256, asked_model=model, model_parameters=model_parameters, n_days=n_days_test)


end_time_testing = time.time()
print(f"Testing : {int((end_time_testing - end_time_training) / 60)} minutes")




