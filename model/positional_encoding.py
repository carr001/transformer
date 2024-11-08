# Author: Vodohleb04
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device, dtype):
        """
        :param d_model: dimensionality of model
        :param max_len: max sequence length
        :param device: torch.Device
        """

        super().__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device, dtype=dtype)  # [max_batch_size, 512]
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device, dtype=dtype)
        pos = pos.unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device, dtype=dtype)
        # 0::2 is mean start:stop:step  from 0 until end (stop is skipped) with step 2
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 0, 2, 4, ... (even numbers)
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))  # 1, 3, 5, ... (odd numbers)


    def forward(self, x):
        _, seq_len = x.size()  # batch_size, seq_len

        return self.encoding[:seq_len, :]
