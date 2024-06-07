import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.skippedAE import skippedAE
from utils.utils import MOTHERDIR

def load_model(model_path, in_channels, out_channels, device):
    model = skippedAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    model.double()
    return model

def generate_ice_data(model, features, feature_index, device):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float64).to(device)
    
    ice_data = np.zeros((features.shape[0], 100))  
    feature_values = np.linspace(features[:, feature_index].cpu().numpy().min(), features[:, feature_index].cpu().numpy().max(), 100)
    
    for i, val in enumerate(feature_values):
        print(f'--- {i} out of {features.shape[0]}', end='\r')
        inputs = features.clone().to(device)
        inputs[:, feature_index] = val
        inputs_tensor = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs_tensor)
        ice_data[:, i] = outputs.cpu().numpy().squeeze()

    return feature_values, ice_data

def plot_ice(feature_values, ice_data, feature_name, outfile):
    plt.figure(figsize=(10, 6))
    
    for i in range(ice_data.shape[0]):
        plt.plot(feature_values, ice_data[i], color='gray', alpha=0.5)
    
    average_ice = np.mean(ice_data, axis=0)
    plt.plot(feature_values, average_ice, color='yellow', linewidth=2)
    sns.rugplot(x=feature_values, color='black', alpha=0.5)
    
    plt.xlabel(feature_name)
    plt.ylabel('Prediction')
    plt.title(f'ICE plot for {feature_name}')
    plt.savefig(outfile)
    plt.close()
