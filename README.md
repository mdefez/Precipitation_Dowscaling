# Precipitation_Dowscaling

The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.

We work with two different datasets, covering different but not exclusive areas, both have 1km and 1hour resolution : the first is the CombiPrecip dataset, the second is the Coméphore dataset, which respectively cover the France and Switzerland area.

Both datasets will be used as input & target :
- The target will be the data as itself
- The input will be the spatially & temporally downsampled data (by two adjustable factors). 

None of the data is stored in this repository, please specify the data path in the corresponding lines. The Coméphore dataset is in open access on Data.gouv.fr

This repository includes three main folders, one for each dataset and the last one stores the topography of both areas. For the moment most of the work is done on Coméphore, the CPC folder might not be up to date.


The UNet folder stores many python files but one should run the CV.py and change the features inside of it to run different models (baselines, UNet with/without attention) with different parameters <br>

