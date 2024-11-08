# Author: Vodohleb04
from math import sqrt
import torch
from torch import nn


class DotProductAttention(nn.Module):

    def __init__(self, d_model, d_head, device, dtype):
        super().__init__()

        self._w_query = nn.Linear(d_model, d_head, bias=False, device=device, dtype=dtype)
        self._w_key = nn.Linear(d_model, d_head, bias=False, device=device, dtype=dtype)
        self._w_value = nn.Linear(d_model, d_head, bias=False, device=device, dtype=dtype)

        self._sqrt_d_head = sqrt(d_head)


    def forward(self, x):
        query = self._w_query(x)
        key = self._w_key(x)
        key.transpose_()
        value = self._w_value(x)

        scores = nn.functional.softmax(
            torch.matmul(query, key) / self._sqrt_d_head
        )

        return torch.matmul(scores, value)

