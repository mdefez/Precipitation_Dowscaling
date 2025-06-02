# The goal of this script is to perform the block Cross Validation over the spatial domain
# This runs a k-fold CV and saves the best performing weights

from dataset import RainSuperResDataset
from train_deterministic import train   
import torch 
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR, CyclicLR
from functools import partial
import torch.nn as nn
from loss import CustomLossTrain, LossTest 
import tools as tool 
from test_deterministic import test 
import time
import pandas as pd
from loss import PercentileDifferenceLoss
from cross_validation_deterministic import k_fold, simple_training

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
temp_factor = 3
spatial_factor = 1
n_inputs = 4        # Frames to take into account as input, the last one is the image to downscale
delta = False        # If we want to predict deltas instead of real frames (except for the first one)

n_days_train = 30 # Only first n_days are used for each month
n_days_test = 2

# Choose wether we want to train/test/both
training = True 
normalizing = False # If False, the code will import the last normalizer saved
testing = True 

cross_val = False # If we want to perform cross validation or simple training

# Training features
batch_size = 48
epochs = 5
learning_rate = 1e-4

# Data directories
input_dir = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/spatial_{spatial_factor}_temp_{temp_factor}'          # Low res frames
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'        # High res targets
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'    # DEM

# Available models. One should choose a model and fill the corresponding parameters
available_model = ["UNet_with_attention", "nearest_neighbor", "bicubic"]
required_model_parameters = {"UNet_with_attention" : ("hard_constraint_mass", "n_inputs", "attention strategy", "number of heads", "window size"),
                             "nearest_neighbor" : [None],
                             "bicubic" : [None]}

# Hard constraint mass
available_strategy_mass = [None, "additive", ("multiplicative", "a function type that operates on tensors")] # The function should apply element wise for tensors
def f_mass(x): # Function to apply element by element to the tensor. Be careful, it should not be zero when x = 0 and i thould not diverge when x is big
    return 1e-3 + x 
treshold_constraint = 5 # Epoch where we should begin to apply conservative transformation

# Attention parameters
list_strat_attention = [["time", "space"], ["space"], ["time"], [None]]     # What type of attention to compute
strat_attention = ["time", "space"]

nb_heads = 4        # Number of attention heads used during the MHA (both for time & space)
if strat_attention == [None]:
    nb_heads = None
window_size = 3     # window size for spatial attention
if "space" not in strat_attention:
    window_size = None

# Choice of the model and parameters
model = "UNet_with_attention" 
model_parameters = (("multiplicative", f_mass), n_inputs, strat_attention, nb_heads, window_size)



# Loss function (used for training
base_loss = nn.MSELoss       # Fill with nn.L1Loss or nn.MSELoss

loss_function = base_loss()

name_loss = {nn.MSELoss : "l2", nn.L1Loss : "l1", PercentileDifferenceLoss : "99th PE"}

# Metric (used for testing)
metric_test = [nn.L1Loss, nn.MSELoss, PercentileDifferenceLoss]       # Fill by the metric to test the model on. The first one will be the main metric (used to compute the best/worst examples)
name_metric = [name_loss[metric] for metric in metric_test]     # Name of the metrics
df_metric = pd.DataFrame({"Name" : name_metric, "Metric" : metric_test})

metric = LossTest(df_metric=df_metric)

# All available scheduler strategies
dict_scheduler = {"Step decay" : ("Step decay", "epoch", partial(StepLR, step_size= 10, gamma=0.2)), # Every step_size epoch, multiply the learning rate by gamma
                  "Cyclical" : ("Cyclical", "batch", partial(CyclicLR, base_lr=learning_rate, max_lr=learning_rate * 10, step_size_up=100, mode='triangular2'))} # The lr goes from min to max in a step_size period. After each cycle, the max value is divided by 2

scheduler = dict_scheduler["Step decay"] # Choose the strategy here

# Normalization strategies
dict_strategies = ["Standard", "min_max", "Robust"]

strat_precip = "min_max"
strat_dem = "min_max"


# Design the name of the run
name_of_the_run = f"deter_alone_{name_loss[base_loss]}_input_{n_inputs}_n_days_{n_days_train}_attention_{strat_attention}_heads_{nb_heads}_delay_constraint_{treshold_constraint}_lr_{learning_rate}_epochs_{epochs}_cross_val_{cross_val}"

if model in ["bicubic", "nearest_neighbor"]: # If the model is not trainable
    name_of_the_run = f"loss_{name_loss[base_loss]}"

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# Check for valid settings
assert not ("time" in strat_attention and n_inputs == 1), "You want to compute temporal attention with a 1-input sequence"
assert not ("time" not in strat_attention and n_inputs != 1), "You want multiple inputs without computing temporal attention"
assert epochs >= treshold_constraint, "The treshold for the mass conservation constraint is set after the number of epochs"
assert model in available_model, "Model not part of available model"
assert strat_attention in list_strat_attention , "Attention strategies not in the list of available attention strategies"
assert len(model_parameters) == len(required_model_parameters[model]), "Wrong number of model parameters filled"
assert not (n_days_test == 1 or n_days_train == 1), "Training or Testing set is empty, don't set n_days to 1"
assert not (training == True and model in ["nearest_neighbor", "bicubic"]), "You are trying to train an untrainable model"
assert not (model in ["nearest_neighbor", "bicubic"] and n_inputs != 1), "You should not set n_inputs to more than 1 if you are using untrainable models"
assert strat_dem in dict_strategies and strat_precip in dict_strategies, "You chose an unavailable scaling strategies"

# CV pipeline 
if training:
    print("Training")
    # Training (Cross validating) dataset
    # Loading all the datasets
    print("Data loading")
    dico_dataset = {}
    for hor in range(4):
        for vert in range(4):
            dico_dataset[f"tile_hor_{hor}_vert_{vert}"] = RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, train=True, n_days=n_days_train, 
                                                                              n_inputs=n_inputs, temp_factor=temp_factor, spatial_factor=spatial_factor, delta=delta)
    print("Data loaded")

    if cross_val == True:
        training_strategy = k_fold
    else:
        training_strategy = simple_training

    weights, loss, best_stats_precip, best_stats_dem = training_strategy(dico_dataset = dico_dataset, normalizing = normalizing, 
                                                                         strat_precip = strat_precip, strat_dem = strat_dem, 
                                                                         batch_size = batch_size, epochs = epochs, scheduler = scheduler, 
                                                                         learning_rate = learning_rate, loss_function = loss_function, 
                                                                         name_of_the_run = name_of_the_run, temp_factor = temp_factor, 
                                                                         spatial_factor = spatial_factor, model = model, 
                                                                         model_parameters = model_parameters, treshold_constraint = treshold_constraint)

    # Save the best model & the associated normalization
    tool.save_model_deter(weights, name_of_the_run, spatial_factor=spatial_factor, temp_factor=temp_factor)
    if normalizing:
        tool.save_transfo(output_path = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/normalization", 
                 best_stats_precip = best_stats_precip, best_stats_dem = best_stats_dem, strat_dem = strat_dem, strat_precip = strat_precip)
    print(f"\nBest model saved with score: {loss}")

end_time_training = time.time()
print(f"Training : {int((end_time_training - start_time) / 60)} minutes")

# Model testing and vizualisation of some plots
if testing:
    print("Testing")
    # Loading the testing dataset
    print("Data loading")
    test_dataset = []
    for hor in range(4):
        for vert in range(4):
            test_dataset.append(RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, temp_factor=temp_factor, 
                                                    train=False, n_days=n_days_test, n_inputs=n_inputs, spatial_factor=spatial_factor, delta = delta))

    test_dataset = ConcatDataset(test_dataset)
    print("Data loaded")
    # Normalize the test dataset, according to the training one
    print("Normalizing data")
    best_transform = tool.load_best_transform(file = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/normalization",
                                         strat_dem=strat_dem, strat_precip=strat_precip)
    transform_precip, transform_dem = best_transform
    normalized_test_dataset = tool.TransformedDataset(base_dataset = test_dataset,
                                                        transform_precip = transform_precip,
                                                        transform_dem = transform_dem)
    print("Data normalized")

    test(test_dataset=normalized_test_dataset, spatial_factor=spatial_factor, temp_factor=temp_factor, name_of_the_run=name_of_the_run,
         criterion=metric, batch_size=batch_size, asked_model=model, model_parameters=model_parameters, delta = delta)


end_time_testing = time.time()
print(f"Testing : {int((end_time_testing - end_time_training) / 60)} minutes")




