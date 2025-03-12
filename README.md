# Precipitation_Dowscaling

The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.

The input data come from the ERA-5 dataset, its resolution is 0.25° and the target data come from the CombiPrecip dataset, its resolution is 1km. Both temporal resolution are 1 hour.

None of the data is stored in this repository, actually it presumes the CombiPreCip dataset (respectively ERA-5) is in the folder : <br>
- ../Data/CPC_file/<CPC19xxxx.h5> <br>
-  ../Data/ERA/ERA5_2019-1_total_precipitation.nc

The librairies used in this repo are stored in the requirements.txt, please note the script generates and compresses pdf files through GhostScript which need to be installed. 

First this repository includes a "Simple baseline folder", where are implemented two shallow models (bicubic interpolation & kNN). To run those models, compute different metrics and plot relevant figures, one can run the "automatisation_all_timestamps_master.py". <br>
Relevant results are stored in the folder Images/, where each subfolder correspond to a timestamp. 