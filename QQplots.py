import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from utils.utils import OFLESDataset, R2Score, trainDataCollecter, testDataCollecter, MOTHERDIR, HEADERS, M1_HEADERS, M4_HEADERS, M3_HEADERS, M2_HEADERS
from utils.ice import generate_ice_data, plot_ice, load_model
import scipy.stats as stats

modelMode = 'MLP' #'WAE' # 
Re = 'R4'
groupName = f'wae_R10{Re[1]}' if modelMode == 'WAE' else f'mlp_R10{Re[1]}'

dt_names = ['M1', 'M2', 'M3', 'M4']

test_org, test_norm, test_means, test_scales = testDataCollecter(Re)
train_org, train_norm, train_means, train_scales = trainDataCollecter(Re)

dir = Path(f'{MOTHERDIR}/Results/{groupName}/')
if not dir.exists():
    dir.mkdir(parents=True, exist_ok=True)

Num = 5000
for dt_name in dt_names:
    print(f'Model Config {dt_name}:')
    dt = test_norm.sample(n=Num, random_state=42).reset_index(drop=True)
    dt = dt.filter(globals()[f"{dt_name}_HEADERS"], axis=1)
    ds = OFLESDataset(dt)
    ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=Num, shuffle=False)

    feature_names = globals()[f"{dt_name}_HEADERS"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f'{MOTHERDIR}/checkpoints/{groupName}_model_{dt_name}.pt'
    model = load_model(modelMode, model_path, ds[0].shape[0] - 1, 1, device)

    for batch_idx, batch in enumerate(ds_loader):
        print(f'Processing batch {batch_idx+1}/{len(ds_loader)}')
        features = batch[:, 0:-1].to(device)
        target = batch[:, -1].to(device)

        # Model output
        model_output = model(features).detach().cpu().numpy().squeeze()

        plt.figure()
        plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica"
            })
        stats.probplot(model_output, dist="norm", plot=plt)
        plt.title(f'QQ Plot for Model Output in {dt_name}')
        plt.savefig(dir / f'{groupName}_{dt_name}_ModelOutput_QQ.png')
        plt.close()
