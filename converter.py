import SPE_Handler_lite as SPE
import os
import glob


def find_spe_files(directory):
    # Construct the search pattern
    search_pattern = os.path.join(directory, '*.spe')

    # Find all .spe files in the directory
    spe_files = glob.glob(search_pattern)

    # Replace backslashes with forward slashes in file paths
    spe_files = [file.replace('\\', '/') for file in spe_files]

    return spe_files


search_directory = "my_data"
save_directory = "save_here"
spe_files = find_spe_files(search_directory)

for spe_file in spe_files:
    SPE.spectra_from_spe(spe_file, save_directory)
