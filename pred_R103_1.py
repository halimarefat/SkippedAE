import numpy as np
import torch
import time
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from tqdm import tqdm

from utils.utils import OFLESDataset, R2Score, HEADERS, M1_HEADERS
from model.skippedAE import skippedAE

Re = 'R3'
groupName = 'R103'

with open(f'datasets/coeffs/test/fieldData_{Re}_unseen_means.txt', 'r') as file:
    data = [float(line.strip()) for line in file]
test_means = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

with open(f'datasets/coeffs/test/fieldData_{Re}_unseen_scales.txt', 'r') as file:
    data = [float(line.strip()) for line in file]
test_scales = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

test_norm = pd.read_csv(f'datasets/normalized/test/fieldData_{Re}_unseen_norm.txt', sep=' ', names=HEADERS)
test_org = pd.read_csv(f'datasets/original/test/fieldData_{Re}_unseen.txt', sep=' ', names=HEADERS)

M1 = test_norm.filter(M1_HEADERS, axis=1)

dt = M1
dt_name = 'M1'


ds = OFLESDataset(dt)

test_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=50000, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = f"./checkpoints/R104_model_{dt_name}.pt"
out_channels = 1
in_channels = dt.shape[1] - out_channels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = skippedAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)  
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)
model.double()
criterion = nn.MSELoss()

Cs_true = []
Cs_pred = []
    
start_time = time.time()
test_loss = 0.0
test_loop = tqdm(test_loader, position=0, leave=True)
for batch in test_loop:
    features = batch[:, :-1].to(device)
    label = batch[:, -1].to(device)
    output = model(features)
    pred = output.squeeze()
    loss = criterion(pred, label)
    test_loss += loss.item()
    test_loop.set_postfix(loss=loss.item())
    Cs_pred.append(pred.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)
    Cs_true.append(label.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)

Cs_true = np.concatenate(Cs_true).ravel()
Cs_pred = np.concatenate(Cs_pred).ravel()

test_loss /= len(test_loader)
test_coefficient = R2Score(label, pred).item()

print(f'loss is {test_loss}, and R2 score is {test_coefficient}')
print(f'shape of Cs_true is {Cs_true.shape}')
print(f'shape of Cs_pred is {Cs_pred.shape}')

n, xedges, yedges = np.histogram2d(Cs_true, Cs_pred, bins=[1500, 1501])
jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

plt.pcolormesh(X, Y, jpdf.T, shading='auto', cmap='jet')
plt.clim([-4, 54])
plt.xlabel(r'$C_s$', fontsize=14)
plt.ylabel(r'$\tilde{C_s}$', fontsize=14)
plt.xlim([-0.15, 0.15])
plt.ylim([-0.15, 0.15])

plt.tight_layout()
plt.savefig(f'./Results/{groupName}/{groupName}_{dt_name}_jpdf.png')
plt.close()

plt.hist(Cs_true, bins=1000, density=True, alpha=0.6, histtype=u'step', color='blue')
plt.hist(Cs_pred, bins=1000, density=True, alpha=0.6, histtype=u'step', color='red')
plt.xlim([-0.3, 0.3])
plt.xlabel(r'$C_s$', fontsize=14)
plt.xlabel('density', fontsize=14)
plt.legend(['GT', 'SkipAE'])
plt.savefig(f'./Results/{groupName}/{groupName}_{dt_name}_density.png')
plt.close()
