# The goal of this file is to extract the product id corresponding to each specified tiles

import pandas as pd
import requests

df_coordinates = pd.read_excel("DEM/correct_format_filename.xlsx")

list_product_id = []
for k in range(len(df_coordinates)):
    print(f"Progress : {100*k / len(df_coordinates)}%")

    filename = df_coordinates.loc[k, "filename to give to the API"]

    req = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{filename}'")
    json = req.json()
    df_temp = pd.DataFrame.from_dict(json['value'])

    if "Id" in df_temp.columns:
        print("found")
        list_product_id.append(df_temp["Id"].loc[0])
    else:
        list_product_id.append(None)

df_coordinates["Id"] = list_product_id

df_coordinates.to_excel("DEM/coordinates_with_id.xlsx")

