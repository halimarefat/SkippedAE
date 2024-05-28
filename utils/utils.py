import sys
import torch
from torch.utils.data import DataLoader, Dataset

sys_epsilon = sys.float_info.epsilon

HEADERS = ["t",                                             # time
           "X", "Y", "Z",                                   # spacial coordinates
           "Ux", "Uy", "Uz",                                # velocity components
           "G1", "G2", "G3", "G4", "G5", "G6",              # velocity gradient tensor components
           "S1", "S2", "S3", "S4", "S5", "S6",              # strain rate tensor compnents
           "UUp1", "UUp2", "UUp3", "UUp4", "UUp5", "UUp6",  # resolved Reynolds stress tensor components
           "Cs"]                                            # Smagorinsky coefficient

M1_HEADERS = ['Ux', 'Uy', 'Uz', 'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 'Cs']
M2_HEADERS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 'Cs']
M3_HEADERS = ['Ux', 'Uy', 'Uz', 'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 'Cs']
M4_HEADERS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 'Cs']
M5_HEADERS = ['Ux', 'Uy', 'Uz', 
              'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 
              'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 
              'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 
              'Cs']

class OFLESDataset(Dataset):
    
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
    
def R2Score(y_true, y_pred):
    SS_res = torch.sum(torch.square(y_true - y_pred))
    SS_tot = torch.var(y_true, unbiased=False) * y_true.size(0)
    return 1 - SS_res / (SS_tot + sys_epsilon)
