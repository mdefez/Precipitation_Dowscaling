# The goal of this file is to explore the Comephore dataset
# Especially, one could want to visualize the distribution to delete the potential outliers

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd

# Import config
from Coméphore.Config import working_directory, data_directory

# This function get all the precip data for one year into a 1D array
# The random variable X is the amount of precipitation that fell over the hour for a pixel (this is computed for every pixel of every frame of the year)
def get_data(year):
    all_arrays = []
    # We divide the work for each tile because of memory issues
    def get_data_tile(year, hor, vert):
        # Set the root folder
        folder_path = data_directory + f'target_data/{year}/tile_hor_{hor}_vert_{vert}/'

        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

        # Collect all the pixels
        all_pixels = []

        for file_name in npy_files:
            file_path = os.path.join(folder_path, file_name)
            array = np.load(file_path)        # Load the array
            all_pixels.append(array.flatten())  # Flatten to 1D

        # Concatenate all pixels into one array
        all_pixels = np.concatenate(all_pixels) # This is a 1D array

        return all_pixels
    
    for hor in range(4):
        for vert in range(4):
            print(hor, vert)
            all_pixels = get_data_tile(year, hor, vert)
            all_arrays.append(all_pixels)

    all_arrays = np.concatenate(all_arrays)

    # Save the array
    np.save(working_directory + f'STVD/Data_analysis/data_{year}.npy', all_arrays)

# Plot the distribution
def vizualize_distribution(year):
    # load the data
    all_pixels = np.load(working_directory + f'STVD/Data_analysis/data_{year}.npy')

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_pixels, bins=100)
    plt.title(f"Distribution of precipitation")
    plt.xlabel("Precipitation over the last hour")
    plt.xscale("log")
    #plt.ylim(top=0.3 * 1e-9)
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(working_directory + "STVD/Data_analysis/distribution.png")

# Fit a Gamma law 
def fit_gamma(year):
    # load the data
    all_pixels = np.load(working_directory + f'STVD/Data_analysis/data_{year}.npy')
    all_pixels = all_pixels[all_pixels > 0]  # remove exact 0s to fit the gamma law
    print(all_pixels.max())
    # Fit Gamma distribution to data
    a, loc, scale = stats.gamma.fit(all_pixels, floc = 0)

    # Create a range of values for plotting the fitted PDF
    x = np.linspace(all_pixels.min(), all_pixels.max(), 1000)
    fitted_pdf = stats.gamma.pdf(x, a, loc=loc, scale=scale)

    # Calculate 99.5% quantile
    quantile_995 = stats.gamma.ppf(0.995, a, loc=loc, scale=scale)
    print(f"99.5% quantile of fitted Gamma distribution: {quantile_995:.4f}")

    for dizaine in range(10, 100, 10):
        quantile = stats.gamma.ppf(dizaine/100, a, loc=loc, scale=scale)
        print(f"{dizaine} quantile of fitted Gamma distribution: {quantile:.4f}")


    # Plot empirical distribution with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(all_pixels, bins=100, stat='density', color='skyblue', label='Empirical data')

    # Plot fitted Gamma PDF
    plt.plot(x, fitted_pdf, 'r-', lw=2, label=f'Fitted Gamma PDF\nshape={a:.2f}, loc={loc:.2f}, scale={scale:.2f}')

    plt.title("Pixel Value Distribution and Fitted Gamma Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(working_directory + "STVD/Data_analysis/gamma_fit.png")

# Side note : The 99.5% quantile s'élève à 55mm/h ce qui est cohérent pour la RNB. Toutes les valeurs supérieures seront ramenées à ce "maximum"


# This function computes the n quantiles of the precipitation. The data we used to compute those quantiles is saved from get_data. 
# Usually, it should be the training data, this way one could save the computed quantiles to plot the PITD.
# Sometimes the quantiles are the same (especially for 0, given that 88% of the data is 0), so we only keep the biggest associated value (88% in the case of 0)
def get_quantile(year, nb_quantiles):

    # load the data
    all_pixels = np.load(working_directory + f'STVD/Data_analysis/data_{year}.npy')

    # Compute the quantiles
    quantile_levels = np.linspace(0.88, 1, nb_quantiles+1)

    quantiles = np.quantile(all_pixels, quantile_levels)

    # Save the quantiles as a df
    df = pd.DataFrame({
        'quantile': quantile_levels,
        'value': quantiles
    })

    # Keep the biggest value 
    df = df.drop_duplicates(subset = ["value"], keep = "last")

    # Insert the 0 and 1 quantile values, mandatory to compute the histogram (actually 1 is already there, we only insert 0)
    new_row = {'quantile': 0, 'value': 0}
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

    # Save and plot the CSV
    df.to_csv(working_directory + f"STVD/Data_analysis/{nb_quantiles}_quantiles.csv", index=False)
    print(df)


# get_data(2023)

get_quantile(year = 2023, nb_quantiles = 8)


