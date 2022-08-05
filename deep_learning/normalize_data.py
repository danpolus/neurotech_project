import numpy as np
import os
from tqdm import tqdm
import random
import shutil
data_path ='../data/datasets/results/numpy/'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')
double_dip_path = os.path.join(data_path, 'double_dip')


def all_training_files():
    for f in os.listdir(train_path):
        yield os.path.join(train_path, f)

def calculate_distribution_for_each_feature(files_to_sample=60):
    # go over all files, and check the per column feature stats
    files = list(all_training_files())
    random.shuffle(files)
    files = files[:files_to_sample]
    distributions = []
    number_of_features = np.load(files[0]).shape[2]
    for feature in tqdm(range(number_of_features)):
        all_feature_data = []
        for f in files:
            all_feature_data.append(np.load(f)[:, :, feature])

        all_feature_data = np.concatenate(all_feature_data)
        distributions.append((np.mean(all_feature_data), np.std(all_feature_data)))

    return distributions


def normalize_data(data:np.ndarray, distributions):
    output = data.copy()
    for feature in range(data.shape[2]):
        mean, std = distributions[feature]
        output[:, :, feature] = (data[: , :, feature] - mean) / (std + 1e-7)

    return output

def normalize_directory(source, dest, distributions):
    for file_name in tqdm(os.listdir(source)):
        data = np.load(os.path.join(source, file_name))
        normalized_data = normalize_data(data, distributions).astype(np.float32)
        np.save(os.path.join(dest, file_name), normalized_data)


def normalize_everything(distributions):
    new_path = '../data/datasets/results/normalized_numpy'
    normalize_directory(train_path, os.path.join(new_path, 'train'), distributions)
    normalize_directory(test_path, os.path.join(new_path, 'test'), distributions)
    normalize_directory(double_dip_path, os.path.join(new_path, 'double_dip'), distributions)


if __name__ == '__main__':
    distributions = calculate_distribution_for_each_feature(60)
    normalize_everything(distributions)