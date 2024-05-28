import torch
import time
import json
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from utils.utils import OFLESDataset, R2Score, HEADERS, M5_HEADERS
from model.skippedAE import skippedAE

Re = 'R4'
groupName = 'R104'

with open(f'datasets/coeffs/train/fieldData_{Re}_seen_means.txt', 'r') as file:
    data = [float(line.strip()) for line in file]
train_means = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

with open(f'datasets/coeffs/train/fieldData_{Re}_seen_scales.txt', 'r') as file:
    data = [float(line.strip()) for line in file]
train_scales = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

train_norm = pd.read_csv(f'datasets/normalized/train/fieldData_{Re}_seen_norm.txt', sep=' ', names=HEADERS)
train_org = pd.read_csv(f'datasets/original/train/fieldData_{Re}_seen.txt', sep=' ', names=HEADERS)

M5 = train_norm.filter(M5_HEADERS, axis=1)

dt = M5
dt_name = 'M5'

learning_rate = 0.001
num_epochs = 500
patience = 40
best_model_path = f'./checkpoints/{groupName}_model_{dt_name}.pt'
out_channels = 1
in_channels = dt.shape[1] - out_channels 
split_sz = 0.8
batch_sz_trn = 4096
batch_sz_val = int(batch_sz_trn / 4)

mask = np.random.rand(len(dt)) < split_sz
train = dt[mask].reset_index(drop=True) 
val = dt[~mask].reset_index(drop=True)

train_dataset = OFLESDataset(train)
val_dataset = OFLESDataset(val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_sz_trn, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_sz_val, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = skippedAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)  
model.to(device)
model.double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=0.2)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_coefficient": [],
    "val_coefficient": [],
    "learning_rates": [],
    "epoch_times": []
}

best_val_loss = np.inf
patience_counter = 0

for epoch in range(num_epochs):
    start_time = time.time()
    
    model.train()
    train_loss = 0.0
    y_train_true = []
    y_train_pred = []
    
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)
    for batch in train_loop:
        inputs = batch[:, 0:-1].to(device)
        target = batch[:, -1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        y_train_true.append(target)
        y_train_pred.append(outputs.squeeze())
        
        train_loop.set_postfix(loss=loss.item())
    
    train_loss /= len(train_loader)
    y_train_true = torch.cat(y_train_true)
    y_train_pred = torch.cat(y_train_pred)
    train_coefficient = R2Score(y_train_true, y_train_pred).item()
    
    model.eval()
    val_loss = 0.0
    y_val_true = []
    y_val_pred = []
    
    val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", leave=False)
    with torch.no_grad():
        for batch in val_loop:
            inputs = batch[:, 0:-1].to(device)
            target = batch[:, -1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), target)
            val_loss += loss.item()
            y_val_true.append(target)
            y_val_pred.append(outputs.squeeze())
            
            val_loop.set_postfix(loss=loss.item())
    
    val_loss /= len(val_loader)
    y_val_true = torch.cat(y_val_true)
    y_val_pred = torch.cat(y_val_pred)
    val_coefficient = R2Score(y_val_true, y_val_pred).item()
    
    scheduler.step()
    epoch_duration = time.time() - start_time
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_coefficient"].append(train_coefficient)
    history["val_coefficient"].append(val_coefficient)
    history["learning_rates"].append(optimizer.param_groups[0]['lr'])
    history["epoch_times"].append(epoch_duration)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train R^2: {train_coefficient:.4f}, Val R^2: {val_coefficient:.4f} -> Saving best model")
    else:
        patience_counter += 1
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train R^2: {train_coefficient:.4f}, Val R^2: {val_coefficient:.4f} -> No improvement")
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

print(f"Training complete. \n Best model saved to '{best_model_path}'.")

with open(f"./logs/{groupName}_training_history_{dt_name}.json", "w") as f:
    json.dump(history, f)
    
print(f"\n Training history saved to './logs/{groupName}_training_history_{dt_name}.json'")

data_iter = iter(train_loader)
next(data_iter)[:,0:-1]

traced_script_module = torch.jit.trace(model, next(data_iter)[:,0:-1].to(device))
traced_script_module.save(f"./traced/{groupName}_traced_model_{dt_name}.pt")

print(f"Traced model saved to './traced/{groupName}_traced_model_{dt_name}.pt'")