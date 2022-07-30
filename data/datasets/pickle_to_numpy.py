import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

def process_spectral_slope(spectral_slope_data):
    return spectral_slope_data.reshape([-1, 1])

def process_power_band(power_band_data):
    power_band_order = ['alpha', 'beta', 'gamma', 'delta', 'theta']
    power_band_outputs = [None, None, None, None, None]
    next_index = 10 # initially, outside of array
    for i in power_band_data:
        if type(i) == str:
            next_index = power_band_order.index(i)

        else:
            power_band_outputs[next_index] = i

    return np.array(power_band_outputs).T


def process_spectra(spectra_data):
    return spectra_data['P']


def process_spectral_entropy(spectral_ent_data):
    return spectral_ent_data.reshape([-1, 1])


def file_data_to_numpy_array(file_data, empty_lz_complex=True):
    per_chunk_data = []
    for i in range(len(file_data['Spectra'])):
        slope = process_spectral_slope(file_data['SpectralSlope'][i])
        band_power = process_power_band(file_data['BandPowerdB'][i])
        spectra = process_spectra(file_data['Spectra'][i])
        ent = process_spectral_entropy(file_data['SpectralEntropy'][i])

        to_cat = [slope, band_power, spectra, ent]
        if empty_lz_complex:
            to_cat.append(np.ones_like(slope))

        per_chunk_data.append(np.concatenate(to_cat, axis=1))

    return np.array(per_chunk_data)


def process_files():
    for file_path in tqdm(os.listdir('./results/dicts')):
        full_path = os.path.join('./results/dicts', file_path)
        with open(full_path, 'rb') as f:
            file_data = pickle.load(f)
            numpy_data = file_data_to_numpy_array(file_data)
            new_path = full_path.replace('dicts', 'numpy')
            np.save(new_path, numpy_data)


if __name__ == '__main__':
    process_files()

