# Author: Vodohleb04
import torch
from torch import nn
from torch.nn.functional import leaky_relu


class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_hidden, device, dtype):
        super().__init__()
        self._linear1 = nn.Linear(d_model, d_hidden, bias=True, device=device, dtype=dtype)
        nn.init.kaiming_normal_(self._linear1.weight)

        self._linear2 = nn.Linear(d_hidden, d_model, bias=True, device=device, dtype=dtype)


    def forward(self, x):
        output = self._linear1(x)
        nn.functional.relu_(output)
        nn.functional.leaky_relu_(output, negative_slope=0.03)
        output = self._linear2(output)
        return output


