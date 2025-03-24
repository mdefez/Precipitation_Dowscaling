from email.errors import StartBoundaryNotFoundDefect
from traceback import print_exc
from xml.sax import SAXNotRecognizedException
import rasterio
import rasterio.warp
import os
import re

# Original projection (Coméphore)

# We define multiple strategies to project the data
comephore_crs = rasterio.crs.CRS.from_wkt('''
PROJCS["unknown",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Polar_Stereographic"],
    PARAMETER["latitude_of_origin",45],
    PARAMETER["central_meridian",0],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["metre",1],
    AXIS["Easting",SOUTH],
    AXIS["Northing",SOUTH]]
''')

epsg_9829 = "EPSG:9829"

epsg_2154 = "EPSG:2154"

crs = comephore_crs # Choose the strategy here

dico_strategy = {comephore_crs : "no_epsg", epsg_2154 : "2154", epsg_9829 : "9829"}

strategy = dico_strategy[crs]

# Only source data from this year will be projected
year_to_reproject = 2019 

# Data are classified by month so we have to loop on them
for k in range(2, 3): # Set to 13 to convert all the data, to 2 just to test on one
    print(k)
    year_month_to_reproject = f"{year_to_reproject}/COMEPHORE_{year_to_reproject}_{k}/{year_to_reproject}"

    # List all files to reproject
    source = f"../../../downscaling/raw_data/Comephore/Original_data/{year_month_to_reproject}"
    all_files = os.listdir(source)
    # We only keep the RR (measurement data)
    all_files = [x for x in all_files if re.search("_RR", x) != None and re.search("xml", x) == None]

    for file in all_files:

        # Load Coméphore data
        with rasterio.open(source + "/" + file) as src:
            # compute the transformation to EPSG:4326
            transform, width, height = rasterio.warp.calculate_default_transform(
                comephore_crs, "EPSG:4326", src.width, src.height, *src.bounds
            )

            # Upload the reprojected data in the destination folder, that we create if it does not exist
            destination = f"../../../downscaling/mdefez/Comephore/Projected_data/test/{strategy}/{year_month_to_reproject}"
            os.makedirs(destination, exist_ok=True)

            with rasterio.open(destination + "/" + "Projected_" + file, "w", driver='GTiff', height=height, width=width,
        count=src.count, dtype=src.dtypes[0], crs="EPSG:4326", transform=transform,
        compress='lzw',  # Compression LZW pour réduire la taille du fichier
        tiled=True,      # Tiling pour améliorer la performance et réduire la taille
        blockxsize=256,  # Taille des blocs pour la compression (ajustable)
        blockysize=256
    ) as dst:
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),  
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=comephore_crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=rasterio.enums.Resampling.nearest
                )


