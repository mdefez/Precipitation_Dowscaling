# This is the main file used to run any architecture
# One can train, test, visualize and compute metrics

# Import librairies
import torch 
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR
from functools import partial
import torch.nn as nn
import time
import pandas as pd
import sys


# Import functions from other files
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic')
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion')
from dataset import RainSuperResDataset
from loss import LossTest, PercentileDifferenceLoss, Log_spectral_distance, EarthMovingDistance, SSIM, PITD
import tools as tool 
from test import test 
from cross_validation import k_fold, simple_training

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
spatial_factor = 10

patience_threshold = 6      # Early training, triggers if there is no improvement for patience_threshold validating epochs
n_inputs = 4                # Ordered frames to take into account as input, the last one is the image to downscale
delta = False               # If we want to predict deltas instead of real frames (except for the first one)

n_scenarios = 3     # Number of scenarios to compute

n_days_train = 28       # Only first n_days are used for each month. Set this to an integer between 2 and 28
n_days_test = 14         # Same for n_test. 

# Choose wether we want to train/test/both and normalize
training = True 
normalizing = False         # If False, the code will import the last normalizer saved
testing = True 

cross_val = False           # If we want to perform cross validation or simple training/validating

# Training features
batch_size = 8
epochs = 120
learning_rate = 1e-4

# Data directories
input_dir = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/spatial_{spatial_factor}_temp_{temp_factor}'          # Low res frames
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'        # High res targets
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'    # DEM

# Available models. One should choose a model and fill the corresponding parameters
available_model_deter = ["UNet_with_attention", "nearest_neighbor", "bicubic"]
required_model_parameters = {"UNet_with_attention" : ("hard_constraint_mass", "n_inputs", "attention strategy", "number of heads", "window_size", 
                                                      "mse_deter", "dir weights deter"),
                             "nearest_neighbor" : [None],
                             "bicubic" : [None]}


##### Attention parameters ##### Shared for both models (except the strategy)

list_strat_attention = [["time", "space"], ["space"], ["time"], [None]]     # What type of attention to compute
strat_attention_diffu = ["time", "space"]
strat_attention_deter = ["time", "space"]

nb_heads = 4                        # Number of attention heads used during the MHA (both for time & space)
window_size = [3, 3, 1, 1, 1]       # window size for spatial attention. Every element should be odd
if "space" not in strat_attention_deter + strat_attention_diffu:
    window_size = [None] * 5


###### Mass conservation ###### Same for both models

available_strategy_mass = [None, ("a function type that operates on tensors", "image or patch scale")] # The function should apply element wise for tensors
# Function to apply element by element to the tensor. Be careful, it should not be zero when x = 0 and it should not diverge when x is big
# It is thus recommended to choose a polynomial with an epsilon for numerical stability
def f_mass(x): 
    return 1e-7 + x**2 
treshold_constraint_deter = 20 # Epoch where we should begin to apply conservative transformation for the deterministic model



##### Deterministic model #####

# For the first n epochs, we add the MSE of the deterministic model (output VS target) in the loss function to force the deter to be decent
lambda_mse = 1
epoch_stop_mse_deter = -1 # Epoch where we should stop adding the MSE to the global loss function. Set to -1 if one doesn't want to use it
mse_deter = (lambda_mse, epoch_stop_mse_deter)

# Load pre train deterministic model
# Fill with the path of the deterministic pre trained UNet one want to use, otherwise None
dir_weights_deter = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/weights/spatial_1_temp_3/deter_alone_l2_input_4_n_days_30_attention_['time', 'space']_heads_4_delay_constraint_5_lr_0.0001_epochs_5_cross_val_False.pth" 
dir_weights_deter = None

# Choice of the model and parameters
model_deter = "UNet_with_attention" 
model_deter_parameters = (("image-scale", f_mass), n_inputs, strat_attention_deter, nb_heads, window_size, mse_deter, dir_weights_deter)

##### Diffusion model #####
use_diffusion = True        # If we want to use diffusion or only the deterministic approach

nb_steps = 1000                         # Number of denoising steps
beta = (0, 0.15, "quadratic")       # beta_start, beta_end, linear/quadratic

conservative_mass_diffusion = ("image-scale", f_mass)     # Scale (image or patch -scale) + Function for the multiplicative approach

model_parameters_diffusion = (nb_steps, beta, conservative_mass_diffusion, nb_heads, window_size, strat_attention_diffu)


# Loss function (used for training)
base_loss = nn.MSELoss       # Loss function computed over velocity or noise. It is highly recommended to use MSE.

# To define specific names for the metrics. All those custom losses are written in Deterministic/loss.py
name_loss = {nn.MSELoss : "l2", nn.L1Loss : "l1", PercentileDifferenceLoss : "99th PE",
             Log_spectral_distance : "Log-spectral distance", EarthMovingDistance : "Earth-Moving Distance",
             SSIM : "SSIM"}

# Metric (used for testing)
metric_test = [base_loss, nn.L1Loss, PercentileDifferenceLoss, Log_spectral_distance, EarthMovingDistance, SSIM]       # Fill by the metric to test the model on. The first one will be the main metric (used to compute the best/worst examples)
name_metric = [name_loss[metric] for metric in metric_test]     # Name of the metrics
df_metric = pd.DataFrame({"Name" : name_metric, "Metric" : metric_test})

metric = LossTest(df_metric=df_metric)

# Scheduler strategies
# Number of steps := number of batch that will be passed into the nn. 
total_steps = epochs * (16 * len(RainSuperResDataset(input_dir, output_dir, channel_dir, 0, 0, train=True, n_days=n_days_train, n_inputs=n_inputs, temp_factor=temp_factor, spatial_factor=spatial_factor, delta=delta))) / batch_size   
dict_scheduler = {"Step decay" : ("Step decay", "epoch", partial(StepLR, step_size= 10, gamma=0.2)), # Every step_size epoch, multiply the learning rate by gamma
                  "Cyclical" : ("Cyclical", "batch", partial(CyclicLR, base_lr=learning_rate, max_lr=learning_rate * 10, step_size_up=100, mode='triangular2')), # The lr goes from min to max in a step_size period. After each cycle, the max value is divided by 2
                    "cosinus decrease" : ("cosinus decrease", "batch", partial(CosineAnnealingLR, T_max=total_steps))} # Cosinus that decreases to 0 in T_max steps 
scheduler = dict_scheduler["cosinus decrease"] # Choose the strategy here


# Normalization strategies
dict_strategies = ["Standard", "min_max", "Robust"]

strat_precip = "min_max"
strat_dem = "min_max"

# Design the name of the run
name_of_the_run = f"diffusion_{use_diffusion}_input_{n_inputs}_n_days_{n_days_train}_{n_days_test}_attention_{strat_attention_deter+strat_attention_diffu}_window_{window_size}_heads_{nb_heads}_delay_constraint_{treshold_constraint_deter}_beta_{beta[0]}_{beta[1]}_lr_{learning_rate}_epochs_{epochs}_cross_val_{cross_val}"

if model_deter in ["bicubic", "nearest_neighbor"]: # If we use a shallow model
    if use_diffusion == False: # If we only use the shallow model
        name_of_the_run = f"{model_deter}_n_days_test_{n_days_test}"

    else:       # If we use diffusion, we want to keep tracks of all features and specify the sallow model we use
        name_of_the_run = f"{model_deter}_" + name_of_the_run


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# Check for valid settings
assert not ("time" in strat_attention_deter+strat_attention_diffu and n_inputs == 1), "You want to compute temporal attention with a 1-input sequence"
assert epochs >= treshold_constraint_deter, "The treshold for the mass conservation constraint is set after the number of epochs"
assert epochs >= epoch_stop_mse_deter, "The treshold for stopping the deterministic MSE in the loss function is set after the number of epochs"
assert model_deter in available_model_deter, "Model not part of available model"
assert n_days_train <= 28, "n_days_train sould be lower than 28 (because of february)"
assert (strat_attention_deter in list_strat_attention) and (strat_attention_diffu in list_strat_attention) , "Attention strategies not in the list of available attention strategies"
assert len(model_deter_parameters) == len(required_model_parameters[model_deter]), "Wrong number of model parameters filled"
assert not (n_days_test == 1 or n_days_train == 1), "Training or Testing set is empty, don't set n_days to 1"
assert not (training == True and (model_deter in ["nearest_neighbor", "bicubic"] and use_diffusion == False)), "You are trying to train an untrainable model"
assert (n_inputs != 1 and "time" in strat_attention_deter+strat_attention_diffu) or n_inputs == 1, "You should not set n_inputs to more than 1 if you don't compute temporal attention"
assert strat_dem in dict_strategies and strat_precip in dict_strategies, "You chose an unavailable scaling strategies"
assert n_scenarios == 1 or use_diffusion == True, "You can't generate multiple scenarios if the model is deterministic"
assert batch_size >= 6 or testing == False, "batch_size can not be lower than 5 during testing (needed to compute best/worst samples)"

print(f"SR factor ({spatial_factor}, {temp_factor})")
print(f"RUN : {name_of_the_run}")

# CV pipeline 
if training:
    print("Training")
    # Training dataset
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

    weights_deter, weights_diffu, loss, best_stats_precip, best_stats_dem = training_strategy(dico_dataset = dico_dataset, normalizing = normalizing, 
                                                                         strat_precip = strat_precip, strat_dem = strat_dem, 
                                                                         batch_size = batch_size, epochs = epochs, scheduler = scheduler, 
                                                                         learning_rate = learning_rate, loss_function = base_loss(), 
                                                                         name_of_the_run = name_of_the_run, temp_factor = temp_factor, 
                                                                         spatial_factor = spatial_factor, model = model_deter, 
                                                                         model_parameters = model_deter_parameters, treshold_constraint_deter = treshold_constraint_deter,
                                                                         n_input = n_inputs, use_diffusion = use_diffusion,
                                                                         model_parameters_diffusion = model_parameters_diffusion, patience_threshold = patience_threshold)

    # Save the best model & the associated normalization
    if model_deter == "UNet_with_attention":
        tool.save_model_deter(weights_deter, name_of_the_run, spatial_factor=spatial_factor, temp_factor=temp_factor)
    if use_diffusion:
        tool.save_model_diffu(weights_diffu, name_of_the_run, spatial_factor=spatial_factor, temp_factor=temp_factor)
    if normalizing:
        tool.save_transfo(output_path = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/normalization", 
                 best_stats_precip = best_stats_precip, best_stats_dem = best_stats_dem, strat_dem = strat_dem, strat_precip = strat_precip)
    print(f"\nModel saved with score: {loss}")

end_time_training = time.time()
print(f"Training : {int((end_time_training - start_time) / 60)} minutes")

# Model testing and vizualisation 
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
         criterion=metric, batch_size=batch_size, asked_model=model_deter, model_parameters=model_deter_parameters, delta = delta, n_inputs = n_inputs,
         model_parameters_diffusion = model_parameters_diffusion, n_scenarios=n_scenarios, use_diffusion = use_diffusion, start_time = start_time)


end_time_testing = time.time()
print(f"Testing : {int((end_time_testing - end_time_training) / 60)} minutes")




