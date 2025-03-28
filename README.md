# Precipitation_Dowscaling

The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.

We work with two different datasets, covering different but not exclusive areas, both have 1km and 1hour resolution : the first is the CombiPrecip dataset, the second is the Compéhore dataset, which respectively cover the France and Switzerland area.

Both datasets will be used as input & target :
- The target will be the data as itself
- The input will be the spatially & temporally downsampled data (by two adjustable factors). It is also possible to spatially blur the data through different filters

None of the data is stored in this repository, please specify the data path in the corresponding lines.

This repository includes two main folders, one for each dataset. Please keep in mind the preprocessing is mostly done on Coméphore, thus the CPC folder might not be up to date. 

Each folder includes a "Simple baseline" for each target dataset, where are implemented two shallow models (bicubic interpolation & kNN). To run those models, compute different metrics and plot relevant figures, one can run the "main.py". <br>
Relevant results are stored in the folder Images, where each subfolder correspond to a timestamp. <br>

