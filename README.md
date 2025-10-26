# Precipitation_Dowscaling

## Scope of the project
The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.

We work with the Coméphore precipitation datasets, covering the french area with a 1km and 1hour resolution.

The dataset will be used as input & target :
- The target will be the data as itself
- The input will be the spatially & temporally coarsened data (by two adjustable factors). 

## Architecture of the repo

This repository includes 2 main folders, one stores the code to train the model on the Coméphore dataset and one stores the topography the corresponding area. 

One can pick the Coméphore/STVD/main.py file and play with all the hyperparameters in the beggining of the file. The model's architecture is composed as follow : 
- First a bicubic interpolation to spatially downscale the data
- Then a deterministic model (in our case a UNet) temporally downscale the data
- Finally, a diffusion model refines the frames

To launch the code, one need to execute the Coméphore/STVD/main.py from the root of the repo.

## Required configuration to use the repo

None of the data is stored in this repository but the Coméphore dataset is in open access on Data.gouv.fr
Please create a Coméphore/config.py file to use this repository and fill it with the following lines :

data_directory =        # Directory where the data is stored
working_directory =     # Directory of the project
original_data_path =    # Directory used to download the original Coméphore data, before any preprocessing
projected_data_path =   # Directory used to store the projected data (in the right EPSG)
topography_directory =  # Directory where the topography is stored

