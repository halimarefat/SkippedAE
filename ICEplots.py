import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from utils.utils import OFLESDataset, HEADERS, trainDataCollecter, testDataCollecter, MOTHERDIR
from model.skippedAE import skippedAE
from utils.ice import generate_ice_data, plot_ice

def load_model(model_path, in_channels, out_channels, device):
    model = skippedAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    model.double()
    return model

Re = 'R4'
groupName = 'R104'

test_org, test_norm, test_means, test_scales = testDataCollecter(Re)
train_org, train_norm, train_means, train_scales = trainDataCollecter(Re)

dt = test_norm.sample(n=50000, random_state=42).reset_index(drop=True)
dt_name = 'M6_ICE'

ds = OFLESDataset(dt)
ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=50000, shuffle=False)

feature_names = HEADERS[:-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f'{MOTHERDIR}/checkpoints/R104_model_M6.pt'
model = load_model(model_path, ds[0].shape[0] - 1, 1, device)


for batch_idx, batch in enumerate(ds_loader):
    print(f'Processing batch {batch_idx+1}/{len(ds_loader)}')
    features = batch[:, 0:-1].to(device)
    target = batch[:, -1].to(device)
    
    for i in range(features.shape[1]):
        print(f'Working on feature {i+1}/{features.shape[1]} in batch {batch_idx+1}')
        feature_values, ice_data = generate_ice_data(model, features, i, device)
        plot_ice(feature_values, ice_data, feature_names[i], f'{MOTHERDIR}/ICEs/{groupName}_{dt_name}_batch{batch_idx+1}_feature{i+1}_.png')
