import sys
import torch
from torch.utils.data import DataLoader, Dataset

sys_epsilon = sys.float_info.epsilon

class MyDataset(Dataset):
    
    def __init__(self, dataframe):
        self.data = dataframe

    def __getitem__(self, index):
        # Ensure all indices are valid
        if index < len(self.data):
            return torch.tensor(self.data.iloc[index].values, dtype=torch.float64)
        else:
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.data)}")

    def __len__(self):
        return len(self.data)