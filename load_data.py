import mat73
import pymatreader
from pymatreader import utils
import h5py
import gc
import time


s = time.time()
gc.disable()
data = pymatreader.read_mat('./coma_dataset_processing/Results_20220728_2125.mat')
# data = mat73.loadmat('./results/Results_20220730_1207.mat')
gc.enable()
e = time.time()
print(e-s)

number_of_files = len(data['Results']['Name'])
import os
import pickle
from tqdm import tqdm
os.makedirs(os.path.join('./results', 'dicts'), exist_ok=True)
for i in tqdm(range(number_of_files), leave=True):
    name = data['Results']['Name'][i]
    new_data = data['Results']['FeaturesExperiment'][i]
    path = os.path.join('./results', 'dicts', name)
    f = open(path, 'wb')
    pickle.dump(new_data, f)
    f.close()