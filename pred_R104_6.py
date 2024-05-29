import numpy as np
import torch
import pandas as pd
from scipy.integrate import trapz

from utils.utils import OFLESDataset, R2Score, HEADERS
from model.skippedAE import skippedAE

Re = 'R4'
groupName = 'R104'

with open(f'datasets/coeffs/test/fieldData_{Re}_unseen_means.txt', 'r') as file:
    data = [float(line.strip()) for line in file]
test_means = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

with open(f'datasets/coeffs/test/fieldData_{Re}_unseen_scales.txt', 'r') as file:
    data = [float(line.strip()) for line in file]
test_scales = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

test_norm = pd.read_csv(f'datasets/normalized/test/fieldData_{Re}_unseen_norm.txt', sep=' ', names=HEADERS)
test_org = pd.read_csv(f'datasets/original/test/fieldData_{Re}_unseen.txt', sep=' ', names=HEADERS)

M6 = test_norm

dt = M6
dt_name = 'M6'

print(f"--- Using this Model Config: {dt_name}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Running on {device}!")

PATH = f"./checkpoints/R104_model_{dt_name}.pt"
out_channels = 1
in_channels = dt.shape[1] - out_channels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = skippedAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)  
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)
model.double()

features = dt[:, :-1].to(device)
label = dt[:, -1].to(device)
pred = model(features)
Cs_norm = pred.detach().cpu().numpy()
Cs_tilde = Cs_norm * test_scales['Cs'].values + test_means['Cs'].values
Cs_GT = label * test_scales['Cs'].values + test_means['Cs'].values

"""
n, xedges, yedges = np.histogram2d(Cs_GT, Cs_tilde.squeeze(), bins=[1500, 1501])
jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

ax = axes[i // 2, i % 2]
c = ax.pcolormesh(X, Y, jpdf.T, shading='auto', cmap='jet')
c.set_clim([-4, 54])
ax.set_title(f'$\mathbf M{dt_name[1]}$', fontsize=16)
ax.set_xlabel(r'$C_s$', fontsize=14)
ax.set_ylabel(r'$\tilde{C_s}$', fontsize=14)
ax.set_xlim([-0.15, 0.15])
ax.set_ylim([-0.15, 0.15])
fig.colorbar(c, ax=ax)

plt.tight_layout()
plt.show()
"""