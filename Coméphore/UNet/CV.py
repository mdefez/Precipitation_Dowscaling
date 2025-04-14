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


# Data directories
input_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data'          # Low res frames
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'        # High res targets
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'    # DEM

# Training features
batch_size = 16
epochs = 20
learning_rate = 1e-5

base_loss = nn.L1Loss()
lambda_conservative = 0.3
lambda_autocorr = 0.3
loss_function = CustomLoss(base_loss=base_loss,
                           lambda_conservative=lambda_conservative,
                           lambda_covariance=lambda_autocorr)

# All available strategies
dict_scheduler = {"Step decay" : ("Step decay", "epoch", partial(StepLR, step_size=10, gamma=0.1)), # Every ste_size epoch, multiply the learning rate by gamma
                  "Cyclical" : ("Cyclical", "batch", partial(CyclicLR, base_lr=1e-5, max_lr=1e-3, step_size_up=50, mode='triangular2'))} # The lr goes from min to max in a step_size period. After each cycle, the max value is divided by 2

scheduler = dict_scheduler["Cyclical"] # Choose the strategy here


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################



# Function that saves the model's weights in the file path
def save_model(weights, filename):
    filepath = '/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/weights/'
    torch.save({'model_state_dict': weights}, filepath + filename)

# Loading all the datasets
dico_dataset = {}
for hor in range(4):
    for vert in range(4):
        dico_dataset[f"tile_hor_{hor}_vert_{vert}"] = RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert,train=False)




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
                          split = k)

    print(f"Loss on split {k+1}: {loss}")

    # Mise à jour du meilleur modèle
    if loss < best_val_score:
        best_val_score = loss
        best_model_state = weights

# Sauvegarde finale du meilleur modèle
save_model(weights, "model_cv.pth")
print(f"\nBest model saved with score: {best_val_score}")






