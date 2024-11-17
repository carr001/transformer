import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, x, y, x_padding_mask, y_padding_mask, device):
        self._x = x
        self._y = y
        self._x_padding_mask = x_padding_mask
        self._y_padding_mask = y_padding_mask
        self._device = device


    def __len__(self):
        return len(self._x)


    def __getitem__(self, idx):
        return (
            self._x[idx].to(self._device),
            self._y[idx].to(self._device),
            self._x_padding_mask[idx].to(self._device),
            self._y_padding_mask[idx].to(self._device)
        )
