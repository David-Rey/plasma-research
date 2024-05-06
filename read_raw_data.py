import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to your file
path = 'raw_data/17_01_19.txt'

# Read the data from the file into a DataFrame
data = pd.read_fwf(path)

# Split metadata from actual data
metadata = data.iloc[:8, :]  # First 8 rows contain metadata


# Extract actual data (excluding metadata)
actual_data = data.iloc[9:, :]
data_list = actual_data.values

# Split each element based on the '\t' delimiter
split_array = np.array([row[0].split('\t') for row in data_list])

# Extract left and right parts
wavelength = split_array[:, 0].astype(float)  # left part
intensity = split_array[:, 1].astype(int)  # right part

plt.plot(wavelength, intensity)
plt.grid(True)
plt.show()

#split_data = [x.split("\t") for x in data_list]






