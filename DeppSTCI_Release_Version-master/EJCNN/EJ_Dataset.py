import torch
from torch.utils.data import Dataset

class EJ_Dataset(Dataset):
    def __init__(self, data, labels, n, m, window):
        self.data = data
        self.labels = labels
        self.n = n
        self.m = m
        self.window = window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index].reshape(4 * self.window + 2, self.n, self.m)
        y = self.labels[index]
        x = torch.tensor(x)
        x = x.float()
        y = torch.tensor(y)
        y = y.float()
        return x, y