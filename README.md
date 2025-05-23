# Precipitation_Dowscaling

The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.

We work with two different precipitation datasets, covering different but not exclusive areas, both have 1km and 1hour resolution : the first is the CombiPrecip dataset from MeteoSwiss, the second is the Coméphore dataset from MeteoFrance, which respectively cover the Switzerland and France lands.

Both datasets will be used as input & target :
- The target will be the data as itself
- The input will be the spatially & temporally coarsened data (by two adjustable factors). 

None of the data is stored in this repository, please specify the data path in the corresponding lines. The Coméphore dataset is in open access on Data.gouv.fr

This repository includes three main folders, one for each dataset and the last one stores the topography of both areas. For the moment most of the work is done on Coméphore, the CPC folder might not be up to date.


One can pick the Coméphore/STVD/main.py file and play with all the hyperparameters in the beggining of the file. The model's architecture is composed as follow : 
- First a bicubic interpolation to spatially downscale the data
- Then a deterministic model (in our case a UNet) temporally downscale the data
- Finally, a diffusion model refines the frames

