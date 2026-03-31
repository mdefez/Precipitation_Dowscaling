def main_function():

    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from Coméphore.Config import working_directory, data_directory
    from Coméphore.Deterministic.dataset import RainSuperResDataset
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from torchsr.models import edsr, ninasr_b2

    import wandb
    wandb.init(project='test', entity='mdefez-cv', name = "training_esdr") 

    EPOCHS = 120
    LR_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VAL_EVERY = 3
    PATIENCE = 3    

    best_val_loss = float("inf")
    no_improve_count = 0

    temp_factor = 1
    spatial_factor = 10

    input_dir = data_directory + f'input_data/spatial_{spatial_factor}_temp_{temp_factor}'          # Low res frames
    output_dir = data_directory + 'target_data'        # High res targets


    dico_dataset = {}
    for hor in range(4):
        for vert in range(4):
            dico_dataset[f"tile_hor_{hor}_vert_{vert}"] = RainSuperResDataset(input_dir, output_dir, channel_root= None, 
                                                                              hor = hor, vert = vert,
                                temp_factor = temp_factor, spatial_factor = spatial_factor, 
                                train=True, n_days = 28)
            
    list_train_dataset = list(dico_dataset.values())[:12]
    train_dataset = ConcatDataset(list_train_dataset)

    list_val_dataset = list(dico_dataset.values())[12:]
    val_dataset = ConcatDataset(list_val_dataset)


    print("Data loaded")

    available_models = {"edsr" : {"model_x3_1" : edsr(scale=3, pretrained=True),
                                  "model_x3_2" : edsr(scale=3, pretrained=True),
                                  "batch_size" : 128,
                                  "scale" : 9,
                                  "hidden_dim" : 56,
                                  "path_weights" : f"benchmark_models/edsr_{spatial_factor}_{temp_factor}.pth"}}
    
    chosen_model = "edsr"

    model_1 = available_models[chosen_model]["model_x3_1"]
    model_2 = available_models[chosen_model]["model_x3_2"]
    BATCH_SIZE = available_models[chosen_model]["batch_size"]
    scale = available_models[chosen_model]["scale"]
    hidden_dim = available_models[chosen_model]["hidden_dim"]

    # Change model settings (RGB as default)
    model_1.head[0] = torch.nn.Conv2d(1, 256, kernel_size=3, padding=1)
    model_1.tail[1] = torch.nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1)

    model_2.head[0] = torch.nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1)
    model_2.tail[1] = torch.nn.Conv2d(256, temp_factor, kernel_size=3, padding=1)

    for model in [model_1, model_2]:
        model.sub_mean = nn.Identity()
        model.add_mean = nn.Identity()
        model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model_1.parameters()) + list(model_2.parameters()), lr=LR_RATE)


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Train
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for lr, _, hr, _ in tqdm(train_dataloader, desc = f"Epoch {epoch}"):
            lr = lr[-1].squeeze(1) # Downscale last frame
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            
            sr_1 = model_1(lr)
            sr_2 = model_2(sr_1)

            output = F.interpolate(
        sr_2,
        scale_factor=spatial_factor/scale,
        mode='bicubic',
        align_corners=False
    )

            loss = criterion(output, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss/len(train_dataloader)
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {average_loss:.6f}")
        wandb.log({"Training loss": average_loss})

        # Validate
        if (epoch - 1) % VAL_EVERY == 0:
            model_1.eval()
            model_2.eval()

            val_loss = 0

            with torch.no_grad():
                for lr, _, hr, _ in tqdm(val_dataloader, desc = "Validating"):
                    lr = lr[-1].squeeze(1) # Downscale last frame
                    lr = lr.to(DEVICE)
                    hr = hr.to(DEVICE)
                    
                    sr_1 = model_1(lr)
                    sr_2 = model_2(sr_1)

                    sr = torch.nn.functional.interpolate(
                        sr_2,
                        scale_factor=spatial_factor/scale,
                        mode='bicubic',
                        align_corners=False
                    )

                    loss = criterion(sr, hr)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            wandb.log({"Validating loss": val_loss})
            print(f"Validation loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0

                # Save best model

                print("model saved")
                torch.save({
                    "model_1": model_1.state_dict(),
                    "model_2": model_2.state_dict()
                }, working_directory + available_models[chosen_model]["path_weights"])


            else:
                no_improve_count += 1

                if no_improve_count >= PATIENCE:
                    print("Early stopping triggered")
                    break


if __name__ == "__main__":
    main_function()
