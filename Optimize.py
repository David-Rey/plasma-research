from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history

import numpy as np
from matplotlib import pyplot as plt

from ReadRamanSpec import ReadRamanSpec
from RamanSpec import RamanSpec


class Optimize:
    def __init__(self, data_path: str, instrument_path: str, B_ev: float, temperature: float, center_wavelength: float,
                 pso_options: dict) -> None:

        self.min_scale = 1E28
        self.max_scale = 3E28
        self.min_phase_shift = -1
        self.max_phase_shift = 1
        self.min_y = 1E4
        self.max_y = 3E4

        self.center_wavelength = center_wavelength

        self.read_raman = ReadRamanSpec(data_path)
        self.raman = RamanSpec(instrument_path, B_ev, temperature)

        self.real_wavelength = self.read_raman.wavelength
        self.real_intensity = self.read_raman.intensity

        self.gen_wavelength = np.linspace(self.read_raman.min_wavelength + self.min_phase_shift,
                                          self.read_raman.max_wavelength + self.max_phase_shift, 1000)
        self.gen_intensity = self.raman.generate_raman(self.gen_wavelength * 1E-9, center_wavelength * 1E-9)

        self.pso_options = pso_options

    def interpolate_intensity(self, real_wavelengths: np.ndarray) -> np.ndarray:
        """
        Interpolates intensity values for a specific target wavelength given known wavelengths and their corresponding intensities.
        """
        return np.interp(real_wavelengths, self.gen_wavelength, self.gen_intensity)

    def objective_func_2d(self, x: np.ndarray, y: float) -> np.ndarray:
        num_partials = np.size(x, 0)
        mse = np.zeros(num_partials)
        for i in range(num_partials):
            mse[i] = self.error_func(x[i, 0], x[i, 1], y)
        return mse

    def objective_func_3d(self, x: np.ndarray) -> np.ndarray:
        num_partials = np.size(x, 0)
        mse = np.zeros(num_partials)
        for i in range(num_partials):
            mse[i] = self.error_func(x[i, 0], x[i, 1], x[i, 2])
        return mse

    def error_func(self, scale, shift, y):
        test_wavelength = self.gen_wavelength + shift
        test_intensity = (self.gen_intensity * scale) + y

        comp_intensity = np.interp(self.real_wavelength, test_wavelength, test_intensity)
        mse = np.mean(comp_intensity - self.real_intensity ** 2)

        return mse

    def optimize_2d(self, y: float, show_cost=False):
        # instantiate the optimizer
        x_min = [self.min_scale, self.min_phase_shift]
        x_max = [self.max_scale, self.max_phase_shift]
        num_particles = 500
        num_iterations = 50

        bounds = (x_min, x_max)

        optimizer = GlobalBestPSO(n_particles=num_particles, dimensions=2, options=self.pso_options, bounds=bounds)

        kwargs = {'y': y}
        cost, pos = optimizer.optimize(self.objective_func_2d, num_iterations, **kwargs)
        if show_cost:
            plot_cost_history(cost_history=optimizer.cost_history)
            plt.grid(True)
        return cost, pos

    def optimize_3d(self, show_cost=False):
        # instantiate the optimizer
        x_min = [self.min_scale, self.min_phase_shift, self.min_y]
        x_max = [self.max_scale, self.max_phase_shift, self.max_y]
        num_particles = 500
        num_iterations = 50

        bounds = (x_min, x_max)
        optimizer = GlobalBestPSO(n_particles=num_particles, dimensions=3, options=self.pso_options, bounds=bounds)

        cost, pos = optimizer.optimize(self.objective_func_3d, num_iterations)
        if show_cost:
            plot_cost_history(cost_history=optimizer.cost_history)
            plt.grid(True)
        return cost, pos

    def draw_overlay(self, scale: float, shift: float, y: float):
        """
        Draws overlay between real and generated intensity values.
        """
        test_wavelength = self.gen_wavelength + shift
        test_intensity = (self.gen_intensity * scale) + y

        comp_intensity = np.interp(self.real_wavelength, test_wavelength, test_intensity)

        plt.figure(figsize=(9, 6))
        plt.plot(self.real_wavelength, self.real_intensity, label='Real')
        plt.plot(self.real_wavelength, comp_intensity, label='Generated')
        plt.xlabel("Wavelength (nm)")
        plt.grid(True)
        plt.legend()

    def draw_contour(self, y: float, pos: tuple):
        # Define the bounds for scale and shift
        x_min = [self.min_scale, self.min_phase_shift]
        x_max = [self.max_scale, self.max_phase_shift]

        # Create a mesh grid for the parameters
        n = 150
        scale_values = np.linspace(x_min[0], x_max[0], n)
        shift_values = np.linspace(x_min[1], x_max[1], n)
        Scale, Shift = np.meshgrid(scale_values, shift_values)

        # Evaluate the objective function at each point in the grid
        Z = np.zeros_like(Scale)
        for i in range(Scale.shape[0]):
            for j in range(Scale.shape[1]):
                Z[i, j] = self.error_func(Scale[i, j], Shift[i, j], y)

        # Plot the contour map
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(Scale, Shift, Z, levels=50, cmap='viridis')
        plt.colorbar(cp)
        plt.plot(pos[0], pos[1], 'ro')
        plt.xlabel('Scale')
        plt.ylabel('Shift')
        plt.grid(True)


if __name__ == '__main__':
    # Define paths
    raw_data_path = 'raw_data/17_05_33.txt'
    instrument_fct_path = 'Fct_instrument/Fct_instrument_1BIN_2400g.csv'

    B_ev = 2.48e-4  # Energy in eV
    T = 250.15  # Temperature in K
    center_wavelength = 532  # Incident light wavelength in nm

    options = {'c1': 1.4, 'c2': 1.4, 'w': 0.7}
    opt = Optimize(raw_data_path, instrument_fct_path, B_ev, T, center_wavelength, options)

    y = opt.read_raman.min_intensity
    cost, pos = opt.optimize_2d(y, show_cost=False)

    scale = pos[0]
    shift = pos[1]

    opt.draw_overlay(scale, shift, y)
    opt.draw_contour(y, pos)
    print(cost)

    plt.show()
