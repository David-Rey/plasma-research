import numpy as np
from matplotlib import pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history

from ReadRamanSpec import ReadRamanSpec
from RamanSpec import RamanSpec


class Optimize:
    """
    A class to optimize parameters for Raman spectrum analysis using Particle Swarm Optimization.
    """

    def __init__(self, data_path: str, instrument_path: str, B_ev: float, temperature: float, center_wavelength: float,
                 pso_options: dict) -> None:
        """
        Initializes the optimization setup with provided parameters and paths.
        """
        self.min_scale = 1E28
        self.max_scale = 5E43
        self.min_phase_shift = -0.8
        self.max_phase_shift = 0.8
        self.min_y = 1E4
        self.max_y = 3E4

        self.center_wavelength = center_wavelength

        self.read_raman = ReadRamanSpec(data_path)
        self.raman = RamanSpec(instrument_path, B_ev, temperature, center_wavelength)

        self.real_wavelength = self.read_raman.wavelength
        self.real_intensity = self.read_raman.intensity

        self.gen_wavelength = np.linspace(self.read_raman.min_wavelength + self.min_phase_shift,
                                          self.read_raman.max_wavelength + self.max_phase_shift, 1000)
        self.raman.generate_raman(1000)
        self.gen_intensity = self.raman.intensity_arr

        self.pso_options = pso_options

        self.priority_function = self.gen_priority_function()

    def gen_priority_function(self) -> np.ndarray:
        """
        generates priority function that multiples with the error. This priorities the peaks of the raman function.
        """
        a1 = 150
        a2 = .15
        priority_function = np.zeros_like(self.real_wavelength)
        for i in range(len(self.real_wavelength)):
            for l in self.raman.peak_lambdas:
                wavelength = self.real_wavelength[i]
                k1 = wavelength - (l * 1E9)
                priority_function[i] += np.exp(-a1 * k1 ** 2)

        for i in range(len(self.real_wavelength)):
            wavelength = self.real_wavelength[i]
            k2 = wavelength - self.center_wavelength
            priority_function[i] *= np.exp(-a2 * k2 ** 2)
        return priority_function

    def objective_func_2d(self, x: np.ndarray, y: float) -> np.ndarray:
        """
        Objective function for 2D optimization to compute mean squared error across multiple particles.
        """
        num_partials = np.size(x, 0)
        mse = np.zeros(num_partials)
        for i in range(num_partials):
            mse[i] = self.error_func(x[i, 0], x[i, 1], y)
        return mse

    def error_func(self, scale, shift, y) -> float:
        """
        Calculates the mean squared error between the generated and real intensity data after applying scale and shift.
        """
        test_wavelength = self.gen_wavelength + shift
        test_intensity = (self.gen_intensity * scale) + y

        comp_intensity = np.interp(self.real_wavelength, test_wavelength, test_intensity)
        diff = comp_intensity - self.real_intensity
        diff_adj = diff * self.priority_function
        mse = np.mean(diff_adj ** 2)

        return mse

    def optimize_2d(self, y: float, show_cost=False):
        """
        Conducts 2D optimization using PSO.
        """
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
        plt.plot([self.real_wavelength[0], self.real_wavelength[-1]], [y, y], '--')
        plt.xlabel("Wavelength (nm)")
        plt.grid(True)
        plt.legend()

    def draw_contour(self, y: float, pos: tuple):
        """
        Draws a contour map of the optimization landscape.
        """
        # Define the bounds for scale and shift
        x_min = [self.min_phase_shift, self.min_scale]
        x_max = [self.max_phase_shift, self.max_scale]

        # Create a mesh grid for the parameters
        n = 150
        shift_values = np.linspace(x_min[0], x_max[0], n)
        scale_values = np.linspace(x_min[1], x_max[1], n)
        Shift, Scale = np.meshgrid(shift_values, scale_values)

        # Evaluate the objective function at each point in the grid
        Z = np.zeros_like(Scale)
        for i in range(Scale.shape[0]):
            for j in range(Scale.shape[1]):
                Z[i, j] = self.error_func(Scale[i, j], Shift[i, j], y)

        # Plot the contour map
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(Shift, Scale, Z, levels=30, cmap='viridis')
        plt.colorbar(cp)
        plt.plot(pos[1], pos[0], 'ro')
        plt.xlabel('Shift')
        plt.ylabel('Scale')
        plt.grid(True)

    def draw_priority_func(self):
        plt.figure(figsize=(9, 6))
        plt.plot(self.real_wavelength, self.priority_function)
        plt.xlabel('Wavelength (nm)')
        plt.title("Priority Function")
        plt.grid(True)


if __name__ == '__main__':
    # Define paths
    raw_data_path = 'raw_data/17_01_19.txt'
    instrument_fct_path = 'Fct_instrument/Fct_instrument_1BIN_2400g.csv'

    B_ev = 2.48e-4  # Energy in eV
    T = 288.15  # Temperature in K
    center_wavelength = 532  # Incident light wavelength in nm

    options = {'c1': 1.4, 'c2': 1.4, 'w': 0.7}
    opt = Optimize(raw_data_path, instrument_fct_path, B_ev, T, center_wavelength, options)

    y = opt.read_raman.min_intensity
    cost, pos = opt.optimize_2d(y, show_cost=False)

    scale = pos[0]
    shift = pos[1]

    opt.draw_overlay(scale, shift, y)
    opt.draw_contour(y, pos)
    #opt.draw_priority_func()

    plt.show()
