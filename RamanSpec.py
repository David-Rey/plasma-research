import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RamanSpec:
    """
    A class to analyze Raman spectra data, calculate Raman intensities, and visualize the spectral data.
    """

    def __init__(self, data_path: str, B_ev: float, temperature: float, max_J=30):
        """
        Initializes the Raman class with data path, rotational constant, temperature, and maximum J value.
        """
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.wavelengths, self.intensities = self.data.iloc[:, 0], self.data.iloc[:, 1]

        self.max_J = max_J
        self.B = B_ev * 1.60218e-19  # Conversion from eV to Joules
        self.T = temperature

        # Constants
        self.h = 6.62607015e-34  # Planck's constant in J*s
        self.c = 299792458  # Speed of light in m/s
        self.k = 1.380649e-23  # Boltzmann's constant in J/K
        self.gamma_squared = 0.505E-48

        # Calculate partition function and instrument function properties
        self.Q = self.partition_function()
        self.center_instrument_function, self.max_instrument_function = self.calculate_center_instrument_func()
        self.n = 1  # Density (could be parameterized or set externally if needed)

    def calculate_center_instrument_func(self):
        """
        Function to calculate the center and max value instrument function.
        """
        max_index = self.intensities.idxmax()  # Get the index of the maximum intensity
        max_value = self.intensities[max_index]  # Get the maximum intensity value using the index
        wavelength = self.wavelengths[max_index]  # Get the wavelength corresponding to the maximum intensity
        return wavelength * 1E-9, max_value

    def interpolate_intensity(self, target_wavelength):
        """
        Interpolates intensity values for a specific target wavelength given known wavelengths and their corresponding intensities.
        """
        return np.interp(target_wavelength * 1e9, self.wavelengths, self.intensities)

    def partition_function(self) -> float:
        """
        Computes the partition function Z to normalize the population distribution.
        """
        Q = 0
        for J in range(self.max_J):
            E_J = self.B * J * (J + 1)
            g_J = 6 if J % 2 == 0 else 3
            Q += g_J * (2 * J + 1) * np.exp(-E_J / (self.k * self.T))
        return Q

    def n_J(self, J: int) -> float:
        """
        Calculate the population of rotational level J.
        """
        E_J = self.B * J * (J + 1)
        g_J = 6 if J % 2 == 0 else 3
        n_J = self.n / self.Q * g_J * (2 * J + 1) * np.exp(-E_J / (self.k * self.T))
        return n_J

    def raman_intensity(self, wavelength: float, lambda_i: float, num_iterations=30) -> float:
        """
        Calculates the Raman intensity for a given wavelength, accounting for both Stokes and anti-Stokes contributions.
        """
        intensity_RM = 0
        for J in range(1, num_iterations):  # Sum over J from 0 to num_iterations
            lambda_J_P2 = lambda_i + ((lambda_i ** 2) / (self.h * self.c)) * self.B * (4 * J + 6)
            lambda_J_M2 = lambda_i - ((lambda_i ** 2) / (self.h * self.c)) * self.B * (4 * J - 2)
            b_J_P2 = (3 * (J + 1) * (J + 2)) / (2 * (2 * J + 1) * (2 * J + 3))
            b_J_M2 = (3 * J * (J - 1)) / (2 * (2 * J + 1) * (2 * J - 1))
            n_J = self.n_J(J)

            lambda_arr = [lambda_J_P2, lambda_J_M2]
            b_j_arr = [b_J_P2, b_J_M2]
            for i in range(2):
                l = lambda_arr[i]  # wavelength
                b_J = b_j_arr[i]

                omega = 1 / (1E2 * l)  # this is omega_0 plus the delta
                d_sigma_d_omega = 64 * np.pi ** 4 * b_J * omega ** 4 * self.gamma_squared

                wavelength_j = wavelength - l
                intensity = self.interpolate_intensity(wavelength_j + self.center_instrument_function)
                intensity_RM += n_J / self.n * d_sigma_d_omega * intensity

        return intensity_RM

    def draw_raman_spectra(self, lambda_nm: float, plot_width=5, num_points=500):
        """
        Plots Raman spectra within a specified wavelength range.
        """
        lambda_i = lambda_nm * 1e-9  # Convert nm to meters

        wavelength_arr = np.linspace(lambda_nm - plot_width, lambda_nm + plot_width,
                                     num_points) * 1e-9  # Wavelength range in meters
        intensity_arr = [self.raman_intensity(w, lambda_i) for w in wavelength_arr]

        plt.figure(figsize=(9, 5))
        plt.plot(wavelength_arr * 1e9, intensity_arr, label='Raman Intensity')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Raman Intensity vs. Wavelength')
        plt.legend()
        plt.grid(True)

    def draw_n_J(self):
        J_values = range(self.max_J)  # Quantum numbers from 0 to max_J
        n_J_values = [self.n_J(J) for J in J_values]  # Population for each J

        plt.figure(figsize=(9, 5))
        plt.plot(J_values, n_J_values, marker='o')
        plt.xlabel('Quantum Number J')
        plt.ylabel('Population nJ')
        plt.title(f'Population Distribution n_J vs. J at T={self.T}K, B={self.B:.2e} J')
        plt.grid(True)

    def draw_intensity(self):
        plt.figure(figsize=(9, 5))
        plt.plot([self.center_instrument_function * 1E9, self.center_instrument_function * 1E9],
                 [0, self.max_instrument_function], '--', color=[0.5, 0.5, 0.5])
        plt.plot(self.wavelengths, self.intensities)
        plt.plot(self.center_instrument_function * 1E9, self.max_instrument_function, marker='o', color='r')

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.grid(True)


if __name__ == "__main__":
    # Experimental parameters
    B_ev = 2.48e-4  # Energy in eV
    T = 288.15  # Temperature in K
    center_wavelength = 532  # Incident light wavelength in nm

    path = 'Fct_instrument/Fct_instrument_1BIN_2400g.csv'
    raman = RamanSpec(path, B_ev, T)

    raman.draw_raman_spectra(center_wavelength, plot_width=4)
    #raman.draw_n_J()
    #raman.draw_intensity()
    plt.show()
