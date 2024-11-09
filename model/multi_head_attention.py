# Author: Vodohleb04
import torch
from torch import nn
from model.dot_product_attention import DotProductAttention


class MultiHeadAttention(nn.Module):

    def _make_mask(self, max_len, device, dtype):
        self._mask = torch.zeros((max_len, max_len), device=device, dtype=dtype, requires_grad=False)
        for i in range(max_len):
            self._mask[i, i+1:] = -torch.inf


    def __init__(self, d_model, d_head, max_len, device, dtype, masked=False, custom_mask=None):
        if d_head > d_model:
            raise ValueError(f"Expected d_head be less than d_model")
        if d_model % d_head != 0:
            raise ValueError(f"Expected d_model to be divisible by d_head without remainder")

        super().__init__()
        heads_amount = int(d_model / d_head)

        self._attention_head_list = nn.ModuleList(
            [DotProductAttention(d_model, d_head, device, dtype) for _ in range(heads_amount)]
        )

        if masked and custom_mask is None:
            self._make_mask(max_len, device, dtype)
        elif custom_mask is not None:
            self._mask = custom_mask
        else:
            self._mask = None

        self._w0 = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)  # concat_heads * w0


    def forward(self, x_query, x_key, x_value):
        head_result_list = [None for _ in range(len(self._attention_head_list))]
        for i, attention_head in enumerate(self._attention_head_list):
            head_result_list[i] = attention_head(x_query, x_key, x_value, mask=self._mask)
        concat_heads = torch.concat(head_result_list, dim=-1)  # heads of corresponding examples from batch are concatenated
        # (head11 + head12 + head13), (head21 + head22 + head23), ...

        multi_head_attention = self._w0(concat_heads)
        return multi_head_attention


if __name__ == "__main__":
    device = torch.device("cuda")
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

    mha = MultiHeadAttention(6, 2, 3, device=device, dtype=dtype, masked=True)
    res = mha(x, x, x)
    res0 = mha(x[0], x[0], x[0])
    print(f"res[0]: {res[0]}")
    print(f"x[0]: {res0}")
    print(res[1])
    print(mha(x[1], x[1], x[1]))


