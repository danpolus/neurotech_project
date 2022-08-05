import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F
from model import MLP
import dataloader
import os
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = False
netdepth = 4
netwidth = 256




double_dip_dataset = dataloader.EEGDataset(os.path.join('../data/datasets/results/normalized_numpy', 'double_dip'), device=device)
number_of_channels = double_dip_dataset[0][0].shape[1]
number_of_features = double_dip_dataset[0][0].shape[2]

model = MLP(depth=netdepth, width=netwidth, use_dropout=dropout, input_channels=number_of_channels * number_of_features, output_ch=7)

path = os.path.join('./logs/cpu_all/49.pt')
model.load_state_dict(torch.load(path))
model.eval()

target_layers = [model.output_layer]


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

all_grads = []

for X, y in double_dip_dataset:
    # zero the parameter gradients
    X.requires_grad = True
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    all_grads.append(X.grad)

all_grads = [i.numpy() for i in all_grads]
grads = np.concatenate(all_grads, axis=0)
grads = abs(grads)

avg = grads.mean(axis=2)
avg = avg.mean(axis=0)

n = 8
ind = np.argpartition(avg, -n)[-n:]
print(ind)