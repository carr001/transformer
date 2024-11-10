#Author: Vodohleb04
import torch
from torch import nn
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward_network import FeedForwardNetwork


class EncoderLayer(nn.Module):

    def __init__(self, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype, custom_mask=None):
        super().__init__()
        self._dropout_prob = dropout_prob

        self._multi_head_attention = MultiHeadAttention(
            d_model, attention_d_head, max_len, device, dtype, custom_mask=custom_mask
        )
        self._attention_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)  # Normalize every token (last dim)

        self._feed_forward_network = FeedForwardNetwork(d_model, ffn_d_hidden, device, dtype)
        self._feed_forward_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)


    def forward(self, x):
        residual = x
        x = self._multi_head_attention(x_query=x, x_key=x, x_value=x)
        x = nn.functional.dropout(x, p=self._dropout_prob, training=self.training)

        x = x + residual
        x = self._attention_norm(x)

        residual = x
        x = self._feed_forward_network(x)
        x = nn.functional.dropout(x, p=self._dropout_prob, training=self.training)

        x = x + residual
        x = self._feed_forward_norm(x)

        return x


class Encoder(nn.Module):

    def __init__(
        self, layers_amount, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype,
        custom_mask=None
    ):
        super().__init__()
        self._encoder_layer_list = nn.ModuleList()
        for _ in range(layers_amount):
            self._encoder_layer_list.append(
                EncoderLayer(d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype, custom_mask)
            )

    def forward(self, x):
        for encoder_layer in self._encoder_layer_list:
            x = encoder_layer(x)
        return x




if __name__ == "__main__":
    enc_l = EncoderLayer(6, 2, 12, 0.1, 3, torch.device("cuda"), torch.float32)

    x = torch.as_tensor(
        [
            [
                [1., 1., 1., 1., 1., 1.],
                [2., 2., 2., 2., 2., 2.],
                [3., 3., 3., 3., 3., 3.]
            ],
            [
                [10., 10., 10., 10., 10., 10.],
                [11., 11., 11., 11., 11., 11.],
                [12., 12., 12., 12., 12., 12.]
            ]
        ], device=torch.device("cuda")
    )

    enc_l(x)
