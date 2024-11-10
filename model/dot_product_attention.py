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
        key_transposed = self._w_key(x_key)  # It will be transposed on the next step
        key_transposed.transpose_(-2, -1)  # Transpose last and pre-last dims (keys of tokens of sequence)
        value = self._w_value(x_value)

        if mask is None:
            scores = nn.functional.softmax(
                torch.matmul(query, key_transposed) / self._sqrt_d_head,
                dim=-1
            )  # Softmax along rows - softmax along scores for every token of query (sum of scores for query token = 1)
        else:
            scores = nn.functional.softmax(
                (torch.matmul(query, key_transposed) / self._sqrt_d_head) + mask[:query.shape[-2], :key_transposed.shape[-1]],
                dim=-1
            )  # Softmax along rows - softmax along scores for every token of query (sum of scores for query token = 1)

        attention = torch.matmul(scores, value)
        return attention

    # query.shape[-2] is the amount of query tokens, key_transposed.shape[-1] is the amount of key tokens
    # (in both batch and one-example inputs)
