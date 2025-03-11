# Precipitation_Dowscaling

The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.
The input's resolution is 0.25° and the target's resolution is 1km, both temporal resolution are 1 hour.

First this repository includes a "Simple baseline folder", where is implemented two shallow models (bicubic interpolation & kNN). To run those models, compute different metrics and plot relevant figures, one can run the "automatisation_all_timestamps_master.py".
Remark : To do so, you must have the CombiPreCip data in the folder : ../Data/CPC_file/<CPC19xxxx.h5> and the ERA-5 2019-01 data in the folder : ../Data/ERA/ERA5_2019-1_total_precipitation.nc