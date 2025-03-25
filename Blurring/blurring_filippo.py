import os
import logging
import glob
import xarray as xr
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_file(file_info):
    """Process a single NetCDF file: Apply Gaussian blur & downsample."""
    try:
        file, cluster_dir, downsampling_factor = file_info

        # Define input & output paths
        data_path = "/Users/fquareng/data"
        input_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold", cluster_dir)
        output_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold_blurred", cluster_dir)
        os.makedirs(output_dir, exist_ok=True)  # Ensure output dir exists

        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".nz", f"_blurred_x{downsampling_factor}.nz"))

        logging.info(f"Processing file: {input_path}")

        # Open NetCDF file
        ds = xr.open_dataset(input_path)

        # Select relevant variables
        selected_vars = ["RELHUM_2M", "T_2M", "PS"]
        ds_selected = ds[selected_vars]

        # Apply Gaussian filter
        def apply_gaussian_filter(data, sigma=2.0):
            return gaussian_filter(data, sigma=sigma, mode="nearest")

        ds_blurred = ds_selected.copy()
        for var in ds_selected.data_vars:
            ds_blurred[var] = xr.DataArray(
                apply_gaussian_filter(ds_selected[var].values, sigma=2.0),
                dims=ds_selected[var].dims,
                coords=ds_selected[var].coords
            )

        # Downsample the data
        downsampled = ds_blurred.isel(
            rlat=slice(0, None, downsampling_factor),
            rlon=slice(0, None, downsampling_factor)
        )

        # Save processed file
        downsampled.to_netcdf(output_path)
        logging.info(f"Successfully processed: {file}")
        return f"Processed: {input_path}"

    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return f"Error processing {file}: {e}"

def main():
    """Main function: Finds clusters, gathers files, and processes them in parallel."""
    # Set up logging
    logging.basicConfig(filename="process_log.log", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    downsampling_factor = 8
    # data_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
    data_path = "/Users/fquareng/data"
    input_root_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold")

    # Find all cluster directories
    cluster_dirs = [d for d in os.listdir(input_root_dir) if os.path.isdir(os.path.join(input_root_dir, d))]

    # Gather all files across clusters
    file_list = []
    for cluster_dir in tqdm(cluster_dirs):
        cluster_path = os.path.join(input_root_dir, cluster_dir)
        input_files = glob.glob(os.path.join(cluster_path, "*.nz"))  # Adjust file extension if needed
        file_list.extend([(os.path.basename(f), cluster_dir, downsampling_factor) for f in input_files])

    logging.info(f"Found {len(file_list)} files across {len(cluster_dirs)} clusters to process.")

    # Process files in parallel using multiprocessing
    num_workers = min(cpu_count(), len(file_list))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, file_list)

    # Print results
    for res in results:
        print(res)

    logging.info("Batch processing completed.")

if __name__ == "__main__":
    main()