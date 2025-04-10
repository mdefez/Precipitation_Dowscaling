# The goal of this file is to design a train function
# It takes as input a training dataset and it trains the model on it 
# It returns the weights and the loss
# It eventually saves the weights

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import UNet
from dataset import RainSuperResDataset
import os 
import matplotlib.pyplot as plt 


learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train the model a,d returns the average loss & the weights
def train(train_dataset, test_dataset, batch_size, epochs, learning_rate = 1e-4, testing = True, saving = False, save_dir = None):

    assert (isinstance(save_dir, str) and save_dir.endswith(".pth") == True) or (saving == False), "Can't save the weights in the specified directory"
    assert testing == False or isinstance(test_dataset, Dataset), "Can't test, test dataset not a torch dataset"
    assert isinstance(train_dataset, Dataset), "Train dataset not a torch dataset"

    # Load the dataset
    print("Data loading")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    try: # One can set no test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    except:
        test_loader = None
    print("Data loaded")

    # Define the model, loss function & optimizer
    model = UNet().to(device)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Training
    print("Training")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        model.train()
        total_loss = 0

        # To compute progress over the epoch
        progress = 0 
        n = train_dataset.__len__()
        
        for inp0, inp1, channel, target in train_loader:
            print(f"Training progress : {100*progress/n}%")
            progress += batch_size

            inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

            output = model(inp0, inp1, channel)     # Compute the output
            loss = criterion(output, target)        # Compute the loss

            optimizer.zero_grad()                   # Set the gradients to 0
            loss.backward()                         # Compute the gradients
            optimizer.step()                        # Update the weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    loss_to_return = avg_loss # If one is only training, the function returns the training loss

    # Test the model on the testing dataset if asked
    if testing == True:
        # Evaluate the model on the testing dataset
        print("Testing")
        model.eval()
        total_test_loss = 0

        # To compute progress over testing
        k = 0 
        n = test_dataset.__len__()
        with torch.no_grad():
            for inp0, inp1, channel, target in test_loader:
                print(f"Testing progress : {100*k/n}%")
                k += batch_size 

                inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

                output = model(inp0, inp1, channel)

                test_loss = criterion(output, target)
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss after Epoch {epoch}: {avg_test_loss:.4f}")
        loss_to_return = avg_test_loss

    # Save the weights if asked
    if saving == True:
        torch.save({'model_state_dict': model.state_dict()}, save_dir)

    return model.state_dict(), loss_to_return


