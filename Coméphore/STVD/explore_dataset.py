# The goal of this file is to explore the Comephore dataset
# Especially, one could want to visualize the distribution to delete the potential outliers

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# This function get all the precip data for one year into a 1D array
# The random variable X is the amount of precipitation that fell over the hour for a pixel (this is computed for every pixel of every frame of the year)
# We only work on one tile because it needs to much memory
def get_data(year):
    # Set the root folder
    folder_path = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data/{year}/tile_hor_2_vert_2/'

    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    # Collect all the pixels
    all_pixels = []

    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        array = np.load(file_path)        # Load the array
        all_pixels.append(array.flatten())  # Flatten to 1D

    # Concatenate all pixels into one array
    all_pixels = np.concatenate(all_pixels) # This is a 1D array

    # Save the array
    np.save('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/all_pixels.npy', all_pixels)

# Plot the distribution
def vizualize_distribution():
    # load the data
    all_pixels = np.load('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/all_pixels.npy')

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_pixels, bins=100)
    plt.title(f"Distribution of precipitation")
    plt.xlabel("Precipitation over the last hour")
    plt.xscale("log")
    #plt.ylim(top=0.3 * 1e-9)
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig("/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/distribution.png")

# Fit a Gamma law 
def fit_gamma():
    # load the data
    all_pixels = np.load('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/all_pixels.npy')
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
    plt.savefig("/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/gamma_fit.png")

get_data(2023)

# Side note : The 99.5% quantile s'élève à 55mm/h ce qui est cohérent pour la RNB. Toutes les valeurs supérieures seront ramenées à ce "maximum"