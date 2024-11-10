#Author: Vodohleb04
from torch import nn

from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForwardNetwork


class DecoderLayer(nn.Module):

    def __init__(
        self, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype, encoder_mask=None
    ):
        super().__init__()
        self._dropout_prob = dropout_prob

        self._masked_self_attention = MultiHeadAttention(d_model, attention_d_head, max_len, device, dtype, masked=True)
        self._self_attention_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

        self._encoder_decoder_attention = MultiHeadAttention(
            d_model, attention_d_head, max_len, device, dtype, custom_mask=encoder_mask
        )
        self._encoder_decoder_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

        self._feed_forward_network = FeedForwardNetwork(d_model, ffn_d_hidden, device, dtype)
        self._feed_forward_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)


    def forward(self, decoder_x, encoder_output):
        residual = decoder_x
        decoder_x = self._masked_self_attention(x_query=decoder_x, x_key=decoder_x, x_value=decoder_x)
        decoder_x = nn.functional.dropout(decoder_x, p=self._dropout_prob, training=self.training)

        decoder_x = decoder_x + residual
        decoder_x = self._self_attention_norm(decoder_x)

        residual = decoder_x
        decoder_x = self._encoder_decoder_attention(x_query=decoder_x, x_key=encoder_output, x_value=encoder_output)
        decoder_x = nn.functional.dropout(decoder_x, p=self._dropout_prob, training=self.training)

        decoder_x = decoder_x + residual
        decoder_x = self._encoder_decoder_norm(decoder_x)

        residual = decoder_x
        decoder_x = self._feed_forward_network(decoder_x)
        decoder_x = nn.functional.dropout(decoder_x, p=self._dropout_prob, training=self.training)

        decoder_x = decoder_x + residual
        decoder_x = self._feed_forward_norm(decoder_x)
        return decoder_x


class Decoder(nn.Module):

    def __init__(
        self, layers_amount, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype,
        encoder_mask=None
    ):
        super().__init__()
        self._decoder_layer_list = nn.ModuleList()

        for _ in range(layers_amount):
            self._decoder_layer_list.append(
                DecoderLayer(
                    d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype, encoder_mask
                )
            )

    def forward(self, decoder_x, encoder_output):
        for decoder_layer in self._decoder_layer_list:
            decoder_x = decoder_layer(decoder_x, encoder_output)
        return decoder_x

