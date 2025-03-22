# Author: Vodohleb04
import torch
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, device, dtype):
        super().__init__()
        if not d_model % num_heads == 0:
            raise ValueError("Expected d_model is divided by num_heads without remainder")

        self._d_head = d_model // num_heads
        self._num_heads = num_heads

        self._w_query = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self._w_key = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self._w_value = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self._w0 = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)


    def dot_product_attention(self, query, key, value, dropout_p=0, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / 1 #self._sqrt_key_d
        if mask is not None:
            scores += mask

        scores = nn.functional.softmax(scores, dim=-1)
        scores = nn.functional.dropout(scores, p=dropout_p, training=self.training)

        attention = torch.matmul(scores, value)
        return attention


    def _batch_forward(self, query_x, key_x, value_x, dropout_p, mask):
        batch_size, seq_len = query_x.size(0), query_x.size(1)

        query = self._w_query(query_x)
        # divide to heads (every head is counted with individual parameters, but in common matrix)
        query = query.view(batch_size, seq_len, self._num_heads, self._d_head)
        # shape (batch_size, num_heads, seq_len, d_head) is needed to make attention across sequence for every head
        query = query.transpose(-3, -2)

        key = self._w_key(key_x)
        key = key.view(batch_size, seq_len, self._num_heads, self._d_head)
        key = key.transpose(-3, -2)

        value = self._w_value(value_x)
        value = value.view(batch_size, seq_len, self._num_heads, self._d_head)
        value = value.transpose(-3, -2)

        multi_head_attention = self.dot_product_attention(query, key, value, dropout_p, mask)
        # return shape to (batch_size, seq_len, num_heads, d_head)
        multi_head_attention = multi_head_attention.transpose(-3, -2).contiguous()
        # return shape to (batch_size, seq_len, d_model)
        multi_head_attention = multi_head_attention.view(batch_size, seq_len, self._num_heads * self._d_head)
        multi_head_attention = self._w0(multi_head_attention)
        return multi_head_attention


    def _no_batch_forward(self, query_x, key_x, value_x, dropout_p, mask):
        seq_len = query_x.size(0)

        query = self._w_query(query_x)
        # divide to heads (every head is counted with individual parameters, but in common matrix)
        query = query.view(seq_len, self._num_heads, self._d_head)
        # shape (num_heads, seq_len, d_head) is needed to make attention across sequence for every head
        query = query.transpose(-3, -2)

        key = self._w_key(key_x)
        key = key.view(seq_len, self._num_heads, self._d_head)
        key = key.transpose(-3, -2)

        value = self._w_value(value_x)
        value = value.view(seq_len, self._num_heads, self._d_head)
        value = value.transpose(-3, -2)

        multi_head_attention = self.dot_product_attention(query, key, value, dropout_p, mask)
        # return shape to (seq_len, num_heads, d_head)
        multi_head_attention = multi_head_attention.transpose(-3, -2).contiguous()
        # return shape to (seq_len, d_model)
        multi_head_attention = multi_head_attention.view(seq_len, self._num_heads * self._d_head)
        multi_head_attention = self._w0(multi_head_attention)
        return multi_head_attention


    def forward(self, query_x, key_x, value_x, dropout_p=0, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(-3)

        if len(query_x.shape) == 2:
            return self._no_batch_forward(query_x, key_x, value_x, dropout_p, mask)
        elif len(query_x.shape) == 3:
            return self._batch_forward(query_x, key_x, value_x, dropout_p, mask)








if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float32
    x = torch.as_tensor(
        [
            [
                [1., 2., 3., 4., 5., 6.],
                [11., 12., 13, 14, 15, 16],
                [21, 22, 23, 24, 25, 26]
            ],
            [
                [31, 32, 33, 34, 35, 36],
                [41, 42, 43, 44, 45, 46],
                [51, 52, 53, 54, 55, 56]
            ]
        ],
        device=device, dtype=dtype
    )

    mha = MultiHeadAttention(6, 2, device=device, dtype=dtype, masked=True)
    res = mha(x, x, x)
    res0 = mha(x[0], x[0], x[0])
    print(f"res[0]: {res[0]}")
    print(f"x[0]: {res0}")
    print(res[1])
    print(mha(x[1], x[1], x[1]))


