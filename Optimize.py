#import pyswarms as ps
import numpy as np

from ReadRamanSpec import ReadRamanSpec
from RamanSpec import RamanSpec


class Optimize:
    def __init__(self, data_path: str, instrument_path: str, B_ev: float, temperature: float, center_wavelength: float):
        self.center_wavelength = center_wavelength

        self.read_raman = ReadRamanSpec(data_path)
        self.raman = RamanSpec(instrument_path, B_ev, temperature)

        self.real_wavelength = self.read_raman.wavelength
        self.real_intensity = self.read_raman.intensity

        self.gen_wavelength = np.linspace(center_wavelength - 5, center_wavelength + 5, 1000)
        self.gen_intensity = self.raman.generate_raman(self.gen_wavelength, center_wavelength * 1E-9)
        print(1)

    def objective_function(self, scale: float, shift: float, y: float):
        print(1)





if __name__ == '__main__':
    # Define paths
    raw_data_path = 'raw_data/17_01_19.txt'
    instrument_fct_path = 'Fct_instrument/Fct_instrument_1BIN_2400g.csv'

    B_ev = 2.48e-4  # Energy in eV
    T = 288.15  # Temperature in K
    center_wavelength = 532  # Incident light wavelength in nm

    opt = Optimize(raw_data_path, instrument_fct_path, B_ev, T, center_wavelength)
