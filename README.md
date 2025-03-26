# Precipitation_Dowscaling

The goal of this project is to apply Video Super-Resolution algorithm to precipitation data.

The input data come from the ERA-5 dataset, its resolution is 0.25° and 1hour.
We have two target data, covering different but not exclusive areas, both have 1km and 1hour resolution : the first is the CombiPrecip dataset, the second is the Compéhore dataset, which respectively cover the France and Switzerland area.

None of the data is stored in this repository, please specify the data path in the corresponding lines.


First this repository includes a "Simple baseline" for each target dataset, where are implemented two shallow models (bicubic interpolation & kNN). To run those models, compute different metrics and plot relevant figures, one can run the "automatisation_all_timestamps_master.py". <br>
Relevant results are stored in the folder Images/, where each subfolder correspond to a timestamp. <br>

For the moment the preprocessing is mostly done on Coméphore, thus the Simple_baseline_CPC might not be up to date. 