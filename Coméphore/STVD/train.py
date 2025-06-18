# The goal of this file is to design a train function
# It takes as input a training dataset and it trains the model on it 
# It returns the weights and the loss
# It returns the weights corresponding to the best model (on the validating set)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, SequentialLR


# Import deterministic model
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic')
from UNet_attention import UNet_with_attention 
from baseline import nearest_neighbor, bicubic
import tools as tool

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
        lambda_mse_deter, epoch_mse_deter = model_parameters[5]     # Strategy for adding (or not) the MSE deterministic into the global loss function
        dir_weights_deter = model_parameters[6]                     # Get the directory of the deterministic's weights if filled
        model_deter = UNet_with_attention(temp_factor=temp_factor, 
                                    spatial_factor=spatial_factor, 
                                    model_parameters=model_parameters).to(device)
        
        if dir_weights_deter != None:       # Load the pretrained deterministic model if asked
            checkpoint = torch.load(dir_weights_deter, map_location=torch.device(device))
            model_deter.load_state_dict(checkpoint['model_state_dict'])
        
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
    scheduler_diff = DiffusionScheduler(timesteps=nb_steps, beta_start=beta[0], beta_end=beta[1], type = beta[2])

    # Define the loss function & optimizer
    criterion = loss_function
    optimizer = optim.Adam(list(model_deter.parameters()) + list(model_diffusion.parameters()), lr=learning_rate)
    scheduler = scheduler(optimizer = optimizer) # We use a scheduler to control the global learning rate

    # We add a warmup to the scheduler to deal with eventual bad weights initialization
    if epoch_batch == "batch":
        warmup_steps = 500
        warmup_scheduler = LinearLR(optimizer = optimizer, start_factor=1e-8 / learning_rate, end_factor=1.0, total_iters=warmup_steps)  # Start with a warm up (increase linearly from 0 to lr)
        scheduler = SequentialLR(optimizer = optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps])  # switch schedulers at warmup_steps

    best_loss = 1e10    # To save the best performing model. We don't save the last one but the one that minimizes the validation loss
    best_weights_deter = None 
    best_weights_diffusion = None

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
            model_input, temporal_embed, t, true_velo = setup_input(device = device, 
                                          scheduler = scheduler_diff, 
                                          A_seq = A_seq, 
                                          C = output_deter, 
                                          B = target, 
                                          temporal_encoder = temporal_encoder)

            pred_velo = model_diffusion(model_input, temporal_embed, t)

            loss = criterion(pred_velo, true_velo)      # Compute the loss (MSE over the velocity)

            if epoch_mse_deter != -1 and epoch_mse_deter <= epoch: # Adapt the loss function to force the deterministic UNet to be decent
                loss += lambda_mse_deter * nn.MSELoss()(output_deter, target)

            optimizer.zero_grad()                   # Set the gradients to 0                
            loss.backward()                         # Compute the gradients
            optimizer.step()                        # Update the weights

            total_loss += loss.item()

            if epoch_batch == "batch":
                scheduler.step() # Tune the learning rate value


        if epoch_batch == "epoch":
            scheduler.step() # Tune the learning rate value

        avg_loss = total_loss / len(train_loader) # Compute the aberage loss over the epoch

        wandb.log({f"Training Loss split {split}": avg_loss}) # Plot the loss on the website
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss}")


        # Validate the model on the validating dataset if asked
        if testing == True and epoch % 3 == 0:
            # Evaluate the model on the validating dataset
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
                    model_input, temporal_embed, t, true_velo = setup_input(device = device, 
                                                scheduler = scheduler_diff, 
                                                A_seq = A_seq, 
                                                C = output_deter, 
                                                B = target, 
                                                temporal_encoder = temporal_encoder)

                    pred_velo = model_diffusion(model_input, temporal_embed, t)

                    test_loss = criterion(pred_velo, true_velo)      # Compute the loss (MSE over the noise)

                    if epoch_mse_deter != -1 and epoch_mse_deter <= epoch: # Adapt the loss function to force the deterministic UNet to be decent
                        test_loss += lambda_mse_deter * nn.MSELoss()(output_deter, target)

                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
    
            # Update weights if best model
            if avg_test_loss <= best_loss:
                best_weights_deter = model_deter.state_dict()
                best_weights_diffusion = model_diffusion.state_dict()

                # Save the ongoing best model 
                tool.save_model_deter(best_weights_deter, name_run, spatial_factor=spatial_factor, temp_factor=temp_factor)
                tool.save_model_diffu(best_weights_diffusion, name_run, spatial_factor=spatial_factor, temp_factor=temp_factor)

                best_loss = avg_test_loss
                patience_counter = 0
            else: # Keep in mind the lack of improvement
                patience_counter += 1

            # Early stop if no significant improve for too long (n testing epochs, so 3*n real epochs)
            patience_treshold = None       # Set to None if one doesn't want any early stopping
            if patience_treshold != None and patience_counter >= patience_treshold:
                return best_weights_deter, best_weights_diffusion, best_loss


            print(f"Validating Loss : {avg_test_loss}")
            wandb.log({"Validating loss": avg_test_loss}) # Plot the loss on the website

    return best_weights_deter, best_weights_diffusion, best_loss



