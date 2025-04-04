# Shallow to script to plot a DEM tile

import rasterio
import matplotlib.pyplot as plt

file_path = "../../../downscaling/mdefez/DEM/data/N46_00_E011/DEM1_SAR_DTE_30_20110517T170701_20140806T170856_ADS_000000_oscI.DEM/Copernicus_DSM_10_N46_00_E011_00/DEM/Copernicus_DSM_10_N46_00_E011_00_DEM.dt2"


with rasterio.open(file_path) as src:
    arr = src.read(1)
    
    print(f"Metadata : {src.profile}")

# Plot the image
plt.imshow(arr, cmap='terrain')  
plt.colorbar(label='Height (m)')  
plt.title(file_path.split("/")[-1].split(".")[0])
plt.savefig("DEM/test_visualization_tile.png")
