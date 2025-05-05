# The goal of this file is to design a train function
# It takes as input a training dataset and it trains the model on it 
# It returns the weights and the loss
# It eventually saves the weights

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from UNet_attention import UNet_with_attention 
from baseline import nearest_neighbor, bicubic
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train the model a,d returns the average loss & the weights
def train(train_dataset, test_dataset, batch_size, epochs, strategy_scheduler, learning_rate, asked_model, model_parameters, 
          temp_factor, spatial_factor, loss_function = nn.L1Loss(), 
          testing = True, saving = False, save_dir = None, split = None, name_run = "run", treshold_constraint = 1):

    assert (isinstance(save_dir, str) and save_dir.endswith(".pth") == True) or (saving == False), "Can't save the weights in the specified directory"
    assert testing == False or isinstance(test_dataset, Dataset), "Can't test, test dataset not a torch dataset"
    assert isinstance(train_dataset, Dataset), "Train dataset not a torch dataset"
    assert isinstance(name_run, str), "The name of the run is not a string"

    name_scheduler, epoch_batch, scheduler = strategy_scheduler # (Name of the schedule, Time where we need to update the scheduler, Scheduler object)

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
        model = UNet_with_attention(temp_factor=temp_factor, 
                                    spatial_factor=spatial_factor, 
                                    model_parameters=model_parameters).to(device)
        
    elif asked_model == "bicubic":
        model = bicubic(temp_factor=temp_factor, spatial_factor=spatial_factor).to(device)

    elif asked_model == "nearest_neighbor":
        model = nearest_neighbor(temp_factor=temp_factor, spatial_factor=spatial_factor).to(device)

    # Define the loss function & optimizer
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = scheduler(optimizer = optimizer) # We use a scheduler to control the global learning rate

    # Training
    print("Training")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        model.train()
        total_loss = 0

        # To compute progress over the epoch
        progress = 0 
        n = train_dataset.__len__()
        
        for list_low_res, channel, target, time_idx in train_loader:
            print(f"Training progress : {100*progress/n:.2f}%")
            progress += batch_size

            channel, target = channel.to(device), target.to(device)
            for k in range(len(list_low_res)):
                list_low_res[k] = list_low_res[k].to(device)
            
            if treshold_constraint >= epoch: # After some epochs, we apply conservative transformation to fine tune the model
                apply_constraint = True

            output = model(list_low_res, channel, apply_constraint = apply_constraint)     # Compute the output
            loss = criterion(output, target)        # Compute the loss

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

    # Test the model on the testing dataset if asked
    if testing == True:
        # Evaluate the model on the testing dataset
        print("Validating")
        model.eval()
        total_test_loss = 0

        # To compute progress over testing
        p = 0 
        n = test_dataset.__len__()
        with torch.no_grad():
            for list_low_res, channel, target, time_idx in test_loader:
                print(f"Validating progress : {100*p/n:.2f}%")
                p += batch_size 

                channel, target = channel.to(device), target.to(device)
                for k in range(len(list_low_res)):
                    list_low_res[k] = list_low_res[k].to(device)

                output = model(list_low_res, channel)

                test_loss = criterion(output, target)
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Validating Loss after Epoch {epoch}: {avg_test_loss}")
        loss_to_return = avg_test_loss

    # Save the weights if asked
    if saving == True:
        torch.save({'model_state_dict': model.state_dict()}, save_dir)

    return model.state_dict(), loss_to_return


