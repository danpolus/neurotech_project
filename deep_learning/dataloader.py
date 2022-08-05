from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import os

classes = {
    'healthy': 0,
    'mcs-': 1,
    'mcs+': 2,
    'emcs': 3,
    'uws': 4,
    'lis': 5,
    'coma': 6
}


class EEGDataset(Dataset):

    def __init__(self, path, device, channels=None):
        """
        path should be a directory that contains normalized numpy files
        """
        self.path = path
        self.files = os.listdir(path)
        self.device = device
        self.channels = None if channels is None else np.array(channels)

    def __len__(self):
        return len(self.files)


    def _get_file_class(self, file_name:str):
        for k in classes.keys():
            if k in file_name.lower():
                return classes[k]

        raise Exception(f'class not found: {file_name}')

    def __getitem__(self, idx):
        file_name = self.files[idx]
        data = np.load(os.path.join(self.path, file_name))
        if self.channels is not None:
            data = data[:, self.channels, :]
        X = torch.tensor(data)
        y = torch.tile(torch.tensor(self._get_file_class(file_name)), [X.shape[0]])
        return X, y