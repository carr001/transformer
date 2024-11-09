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


    def forward(self, x_query, x_key, x_value, mask=None):
        query = self._w_query(x_query)
        key = self._w_key(x_key)
        key.transpose_(-2, -1)
        value = self._w_value(x_value)

        if mask is None:
            scores = nn.functional.softmax(
                torch.matmul(query, key) / self._sqrt_d_head,
                dim=-1
            )
        else:
            scores = nn.functional.softmax(
                (torch.matmul(query, key) / self._sqrt_d_head) + mask,
                dim=-1
            )
        print(scores)
        attention = torch.matmul(scores, value)
        return attention
