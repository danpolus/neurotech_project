import train
import random
import csv
import os
import numpy as np
from tqdm import tqdm

default_run_string = '--disable_neptune --log_dir {log_dir} --data_dir {data_dir} --measure_accuracy_epoch 5 ' \
                     '--save_model_epoch 300 --epochs 50 --expname automation --netdepth 2 --netwidth 100 --dropout --channels {channels}'


def run(log_dir, data_dir, channels):
    run_str = default_run_string.format(log_dir=log_dir, data_dir=data_dir, channels=' '.join([str(ch) for ch in channels]))
    return train.train(run_str, tqdm_monitor=False)

def automate(output_csv_path, data_dir='./normalized_numpy', reruns=1, channels=[1, 2, 4, 6, 8]):
    total_number_of_channels = 137
    log_dir = './log_dir'
    os.makedirs(log_dir, exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['channels', 'double_dip', 'test'])
        writer.writeheader()
        for chan in tqdm(channels, leave=True):
            double_dips = []
            tests = []
            for i in range(reruns):
                current_channels = random.sample(range(total_number_of_channels), k=chan)
                double_dip, test = run(log_dir=log_dir, data_dir=data_dir, channels=current_channels)
                print(f'double_dip {double_dip} test {test}')
                double_dips.append(double_dip)
                tests.append(test)

            writer.writerow({
                'channels': chan,
                'double_dip': np.array(double_dips).mean(),
                'test': np.array(tests).mean()
            })

            f.flush()

if __name__ == '__main__':
    automate('./a.csv', data_dir='../data/datasets/results/normalized_numpy')