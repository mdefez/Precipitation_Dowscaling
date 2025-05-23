# The goal of this file is to design a train function
# It takes as input a training dataset and it trains the model on it 
# It returns the weights and the loss
# It eventually saves the weights if asked

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from tqdm import tqdm

# Import deterministic model
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic')
from UNet_attention import UNet_with_attention 
from baseline import nearest_neighbor, bicubic

# Import diffusion model
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion')
from diffusion_model import UNetforDiffusion, TemporalEncoder, DiffusionScheduler
from tools_diffu import setup_input, bicubic_A_seq

import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train the model a,d returns the average loss & the weights
def train(train_dataset, test_dataset, batch_size, epochs, strategy_scheduler, learning_rate, asked_model, model_parameters, 
          temp_factor, spatial_factor, n_input, loss_function, treshold_constraint_deter, model_parameters_diffusion,
          testing = True, saving = False, save_dir = None, split = None, name_run = "run"):

    assert (isinstance(save_dir, str) and save_dir.endswith(".pth") == True) or (saving == False), "Can't save the weights in the specified directory"
    assert testing == False or isinstance(test_dataset, Dataset), "Can't test, test dataset not a torch dataset"
    assert isinstance(train_dataset, Dataset), "Train dataset not a torch dataset"
    assert isinstance(name_run, str), "The name of the run is not a string"

    name_scheduler, epoch_batch, scheduler = strategy_scheduler # (Name of the schedule, Time where we need to update the scheduler, Scheduler object)

    apply_constraint_deter = False           # Initialize with False
    apply_constraint_diffusion = False          # Initialize with False

    # To plot the loss on WandB, the run name stores the training features
    wandb.init(project='test', entity='mdefez-cv', name = name_run) 

    # Load the dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    try: # One can set no test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    except:
        test_loader = None

    # Define the model, loss function & optimizer
    if asked_model == "UNet_with_attention":
        model_deter = UNet_with_attention(temp_factor=temp_factor, 
                                    spatial_factor=spatial_factor, 
                                    model_parameters=model_parameters).to(device)
        
    elif asked_model == "bicubic":
        model_deter = bicubic(temp_factor=temp_factor, spatial_factor=spatial_factor).to(device)

    elif asked_model == "nearest_neighbor":
        model_deter = nearest_neighbor(temp_factor=temp_factor, spatial_factor=spatial_factor).to(device)

    # Define the diffusion model, the encoding strategy & the noise scheduler
    in_channels = 2*(temp_factor) + 1       # To compute useful dimensions

    nb_steps, beta, conservative_mass_diffusion = model_parameters_diffusion

    model_diffusion = UNetforDiffusion(in_channels=in_channels, base_channels=64, embed_dim=256, time_emb_dim = 128, 
                                       temp_factor = temp_factor, spatial_factor = spatial_factor).to(device)

    temporal_encoder = TemporalEncoder(input_channels=1, embed_dim=256, seq_len=n_input).to(device).train()
    scheduler_diff = DiffusionScheduler(timesteps=nb_steps, beta_start=beta[0], beta_end=beta[1])

    # Define the loss function & optimizer
    criterion = loss_function
    optimizer = optim.Adam(list(model_deter.parameters()) + list(model_diffusion.parameters()), lr=learning_rate)
    scheduler = scheduler(optimizer = optimizer) # We use a scheduler to control the global learning rate

    # Training
    print("Training")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        model_deter.train()
        model_diffusion.train()

        total_loss = 0

        # Loop over the batches        
        for list_low_res, channel, target, time_idx in tqdm(train_loader, desc="Training", leave=False):

            channel, target = channel.to(device), target.to(device)
            for k in range(len(list_low_res)):
                list_low_res[k] = list_low_res[k].to(device)
            
            if treshold_constraint_deter >= epoch: # After some epochs, we apply conservative transformation to fine tune the model
                apply_constraint_deter = True


            # Compute the output of the deterministic model
            output_deter = model_deter(list_low_res, channel, apply_constraint = apply_constraint_deter)     # Compute the output

            ### Compute the output of the diffusion model
            A_seq = bicubic_A_seq(list_low_res)     # Compute the HR from the LR to pass the diffusion UNet
            # Pass the input as the right format for the diffusion model
            model_input, temporal_embed, t, noise = setup_input(device = device, 
                                          scheduler = scheduler_diff, 
                                          A_seq = A_seq, 
                                          C = output_deter, 
                                          B = target, 
                                          temporal_encoder = temporal_encoder)

            pred_noise = model_diffusion(model_input, temporal_embed, t)

            loss = criterion(pred_noise, noise)      # Compute the loss (MSE over the noise)

            optimizer.zero_grad()                   # Set the gradients to 0                
            loss.backward()                         # Compute the gradients
            optimizer.step()                        # Update the weights



            total_loss += loss.item()

            if epoch_batch == "batch":
                scheduler.step() # Tune the learning rate value


        if epoch_batch == "epoch":
            scheduler.step() # Tune the learning rate value

        avg_loss = total_loss / len(train_loader) # Compute the aberage loss over the epoch

        wandb.log({f"Loss split {split}": avg_loss}) # Plot the loss on the website
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss}")

    loss_to_return = avg_loss # If one is only training, the function returns the training loss

    ####################################################################################################################################################
    # N'EST PAS A JOUR ################################################################################################################################
    ####################################################################################################################################################

    # Test the model on the testing dataset if asked
    if testing == True:
        # Evaluate the model on the testing dataset
        print("Validating")
        model_deter.eval()
        model_diffusion.eval()
        total_test_loss = 0

        # To compute progress over testing
        with torch.no_grad():
            for list_low_res, channel, target, time_idx in tqdm(test_loader, desc="Testing"):

                channel, target = channel.to(device), target.to(device)
                for k in range(len(list_low_res)):
                    list_low_res[k] = list_low_res[k].to(device)

                # Compute the output of the deterministic model
                output_deter = model_deter(list_low_res, channel, apply_constraint = True)    

                ### Compute the output of the diffusion model
                A_seq = bicubic_A_seq(list_low_res)     # Compute the HR from the LR to pass the diffusion UNet
                # Pass the input as the right format for the diffusion model
                model_input, temporal_embed, t, noise = setup_input(device = device, 
                                            scheduler = scheduler_diff, 
                                            A_seq = A_seq, 
                                            C = output_deter, 
                                            B = target, 
                                            temporal_encoder = temporal_encoder)

                pred_noise = model_diffusion(model_input, temporal_embed, t)

                test_loss = criterion(pred_noise, noise)      # Compute the loss (MSE over the noise)

                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Validating Loss : {avg_test_loss}")
        loss_to_return = avg_test_loss

    return model_deter.state_dict(), model_diffusion.state_dict(), loss_to_return



