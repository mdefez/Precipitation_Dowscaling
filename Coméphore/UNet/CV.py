# The goal of this script is to perform the block Cross Validation over the spatial domain
# This runs a k-fold CV and saves the best performing weights

from dataset import RainSuperResDataset
from train import train 
import torch 
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR, CyclicLR
from functools import partial
import torch.nn as nn
from loss import CustomLoss
 
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
n_inputs = 6        # Frames to take into account as input

name_of_the_run = "cv_hard_attention_l2" + f"spatial_{spatial_factor}_temp_{temp_factor}_input_{n_inputs}"

# Data directories
input_dir = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/spatial_{spatial_factor}_temp_{temp_factor}'          # Low res frames
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'        # High res targets
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'    # DEM

# Available models
available_model = ["UNet", "UNet_with_attention"]
required_model_parameters = {"UNet" : ("hard_constraint"), "UNet_with_attention" : ("hard_constraint_mass", "n_inputs")}

# Hard constraint mass
available_strategy_mass = [None, "additive", ("multiplicative", "a function type that operates on tensors")] # The function should apply element wise for tensors
def f_mass(x): # Function to apply element by element to the tensor. Be careful, it should not be zero when x = 0 and i thould not diverge when x is big
    return (x+1)**2

# Choice of the model and parameters
model = "UNet_with_attention"
model_parameters = (("multiplicative", f_mass), n_inputs)

# Training features
batch_size = 128
epochs = 10
learning_rate = 1e-5

# Loss function
base_loss = nn.MSELoss()
conservative = False        # If we add the conservative mass soft constraint
lambda_conservative = 0.3
autocorrel = False          # If we add the autocorrelation soft constraint
lambda_autocorr = 0.3

loss_function = CustomLoss(base_loss=base_loss,
                           lambda_conservative=lambda_conservative,
                           conservative=conservative,                  
                           covariance=autocorrel, 
                           lambda_covariance=lambda_autocorr)

# All available strategies
dict_scheduler = {"Step decay" : ("Step decay", "epoch", partial(StepLR, step_size= epochs // 3, gamma=0.1)), # Every ste_size epoch, multiply the learning rate by gamma
                  "Cyclical" : ("Cyclical", "batch", partial(CyclicLR, base_lr=1e-5, max_lr=1e-3, step_size_up=50, mode='triangular2'))} # The lr goes from min to max in a step_size period. After each cycle, the max value is divided by 2

scheduler = dict_scheduler["Step decay"] # Choose the strategy here


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################



# Function that saves the model's weights in the file path
def save_model(weights):
    filepath = '/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/weights/'
    torch.save({'model_state_dict': weights}, filepath + name_of_the_run + ".pth")

# Loading all the datasets
assert n_inputs == 1 or model != "UNet", "You chose multiple inputs for usual UNet"
dico_dataset = {}
for hor in range(4):
    for vert in range(4):
        dico_dataset[f"tile_hor_{hor}_vert_{vert}"] = RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, train=True, n_days=5, n_inputs=n_inputs, 
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
for k in range(len(splits)):
    print(f"Training of the split : {k+1}")
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

    weights, loss = train(train_dataset=train_dataset, 
                          test_dataset=val_dataset, 
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
                          model_parameters=model_parameters)

    print(f"Loss on split {k+1}: {loss}")
    save_model(weights)     # Save model at each split (so that we eventually don't have to wait the end of the CV). To be deleted.

    # Mise à jour du meilleur modèle
    if loss < best_val_score:
        best_val_score = loss
        best_model_state = weights

# Sauvegarde finale du meilleur modèle
save_model(weights)
print(f"\nBest model saved with score: {best_val_score}")






