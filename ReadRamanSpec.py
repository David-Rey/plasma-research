import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ReadRamanSpec:
    """
    A class to read and process Raman spectroscopy data from a specified file.
    """
    def __init__(self, path: str, shift=True):
        """
        Initializes the ReadRamanSpec object by reading and processing the data from the given file path.
        """
        self.path = path
        self.metadata, self.wavelength, self.intensity = self.read_data()
        if shift:
            self.wavelength = self.wavelength + 2
        self.max_intensity = np.max(self.intensity)
        self.min_intensity = np.min(self.intensity)
        self.min_wavelength = np.min(self.wavelength)
        self.max_wavelength = np.max(self.wavelength)
        self.wavelength_delta = self.wavelength[1] - self.wavelength[0]
        print(self.wavelength_delta)

    def read_data(self):
        """
        Reads the spectroscopy data from the file and extracts metadata, wavelength, and intensity.
        """
        try:
            data = pd.read_fwf(self.path)

            # Assume metadata is in the first 8 rows
            metadata = data.iloc[:8, :]

            # Extract actual data starting from the 9th row
            actual_data = data.iloc[9:, :]
            data_list = actual_data.values

            split_data = np.array([row[0].split('\t') for row in data_list])

            # Convert data to appropriate types
            wavelength = split_data[:, 0].astype(float)  # left part
            intensity = split_data[:, 1].astype(float)  # right part

            return metadata, wavelength, intensity
        except Exception as e:
            raise IOError(f"An error occurred while reading the data from {self.path}: {e}")

    def draw_spectrum(self):
        """
        Draws the spectrum on the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.wavelength, self.intensity)
        plt.xlabel('Wavelength (nm)')
        plt.grid(True)


if __name__ == "__main__":
    path = 'raw_data/17_05_33.txt'
    spec = ReadRamanSpec(path)

    spec.draw_spectrum()
    plt.show()
