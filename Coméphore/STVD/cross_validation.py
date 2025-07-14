# This script aims to perform cross_validation or simple training, depending on the user's choice
# Cross validation is a 4-fold over the spatial tiles 
# Simple training uses all the data as training data (no validation)

from train import train
from torch.utils.data import ConcatDataset
import tools as tool 

# Main function that trains the model with the specified training dataset
def main(train_dataset, val_dataset, normalizing, strat_precip, strat_dem,
        batch_size, epochs, scheduler, learning_rate, loss_function, k, name_of_the_run, 
        temp_factor, spatial_factor, model, model_parameters, treshold_constraint_deter, testing, n_input, use_diffusion,
        model_parameters_diffusion, patience_threshold):
    
    # Normalize the data according to the specified strategies

    if normalizing: # Compute the normalizer
        (transform_precip, transform_dem), (stats_precip, stats_dem) = tool.compute_transformation(train_dataset=train_dataset, strat_precip = strat_precip, strat_channel = strat_dem)
    else: # Load the normalizer
        best_transform = tool.load_best_transform(file = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/normalization",
                                        strat_dem=strat_dem, strat_precip=strat_precip)
        stats_precip, stats_dem = None, None
        transform_precip, transform_dem = best_transform

    normalized_train_dataset = tool.TransformedDataset(base_dataset = train_dataset,
                                                    transform_precip = transform_precip,
                                                    transform_dem = transform_dem)
    normalized_val_dataset = tool.TransformedDataset(base_dataset = val_dataset,
                                                    transform_precip = transform_precip,
                                                    transform_dem = transform_dem)
    print("Data normalized")
    weights_deter, weights_diffu, loss = train(train_dataset=normalized_train_dataset, 
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
                        treshold_constraint_deter=treshold_constraint_deter,
                        testing = testing,
                        n_input = n_input,
                        use_diffusion = use_diffusion,
                        model_parameters_diffusion = model_parameters_diffusion,
                        patience_threshold = patience_threshold)
    
    return weights_deter, weights_diffu, loss, stats_precip, stats_dem



# Performs the k-fold cross validation
def k_fold(dico_dataset, normalizing, strat_precip, strat_dem, batch_size, epochs, scheduler, learning_rate, 
                    loss_function, name_of_the_run, temp_factor, spatial_factor, model, model_parameters, treshold_constraint_deter, n_input, use_diffusion,
                     model_parameters_diffusion, patience_threshold):

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

        weights_deter, weights_diffu, loss, stats_precip, stats_dem = main(train_dataset, val_dataset, normalizing, strat_precip, strat_dem,
                                                      batch_size, epochs, scheduler, learning_rate, loss_function, k, name_of_the_run, 
                                                      temp_factor, spatial_factor, model, model_parameters, treshold_constraint_deter, n_input = n_input,
                                                      testing = True, use_diffusion = use_diffusion, 
                                                      model_parameters_diffusion = model_parameters_diffusion, patience_threshold = patience_threshold)

        print(f"Loss on split {k+1}: {loss}")

        # Keep in memory the best model
        if loss < best_val_score:
            best_val_score = loss
            best_weights_deter, best_weights_diffu = weights_deter, weights_diffu

    return best_weights_deter, best_weights_diffu, best_val_score, stats_precip, stats_dem




# Performs simple training
def simple_training(dico_dataset, normalizing, strat_precip, strat_dem, batch_size, epochs, scheduler, learning_rate, 
                    loss_function, name_of_the_run, temp_factor, spatial_factor, model, model_parameters, treshold_constraint_deter, n_input, use_diffusion, 
                     model_parameters_diffusion, patience_threshold):

    list_train_dataset = list(dico_dataset.values())[:12]
    train_dataset = ConcatDataset(list_train_dataset)

    list_val_dataset = list(dico_dataset.values())[12:]
    val_dataset = ConcatDataset(list_val_dataset)

    weights_deter, weights_diffu, loss, stats_precip, stats_dem = main(train_dataset, val_dataset, normalizing, strat_precip, strat_dem,
                                                      batch_size, epochs, scheduler, learning_rate, loss_function, None, name_of_the_run, 
                                                      temp_factor, spatial_factor, model, model_parameters, treshold_constraint_deter, 
                                                      n_input = n_input, testing = True, use_diffusion = use_diffusion,
                                                      model_parameters_diffusion = model_parameters_diffusion, patience_threshold = patience_threshold)

    return weights_deter, weights_diffu, loss, stats_precip, stats_dem