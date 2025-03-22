#Author: Vodohleb04
import torch
from torch import nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward_network import FeedForwardNetwork


class DecoderLayer(nn.Module):

    def __init__(self, d_model, attention_d_head, ffn_d_hidden, dropout_prob, device, dtype):
        super().__init__()
        self._dropout_prob = dropout_prob

        self._masked_self_attention = MultiHeadAttention(d_model, attention_d_head, device, dtype)
        self._self_attention_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

        self._encoder_decoder_attention = MultiHeadAttention(d_model, attention_d_head, device, dtype)
        self._encoder_decoder_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

        self._feed_forward_network = FeedForwardNetwork(d_model, ffn_d_hidden, device, dtype)
        self._feed_forward_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)


    def forward(self, decoder_x, encoder_output, self_attention_mask, encoder_decoder_mask=None):
        residual = decoder_x
        decoder_x = self._masked_self_attention(
            query_x=decoder_x, key_x=decoder_x, value_x=decoder_x, mask=self_attention_mask
        )
        decoder_x = nn.functional.dropout(decoder_x, p=self._dropout_prob, training=self.training)

        decoder_x = decoder_x + residual
        decoder_x = self._self_attention_norm(decoder_x)

        residual = decoder_x
        decoder_x = self._encoder_decoder_attention(
            query_x=decoder_x, key_x=encoder_output, value_x=encoder_output, mask=encoder_decoder_mask
        )
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

    @staticmethod
    def _init_look_ahead_mask(max_len, device, dtype):
        _look_ahead_mask = torch.zeros((max_len, max_len), device=device, dtype=dtype, requires_grad=False)
        for i in range(max_len):
            _look_ahead_mask[i, i + 1:] = -torch.inf
        return _look_ahead_mask

    @staticmethod
    def _count_encoder_decoder_mask(encoder_padding_mask, decoder_x_shape):
        # In query - len of target sequence, in key - len of source sequence. Mask is Query x Key_transpose -> Mask
        #   has size [batch_size?, len_of_target, len_of_source]. decoder_x.shape[-2] is amount of query tokens
        # encoder_padding_mask.shape[-1] is amount of key tokens
        if len(encoder_padding_mask.shape) == 3:
            if encoder_padding_mask.shape[-1] >= decoder_x_shape[-2]:
                encoder_decoder_mask = encoder_padding_mask[:, :decoder_x_shape[-2], :]
            else:
                encoder_decoder_mask = encoder_padding_mask[:,:1,:]  # mask for token with index 0 in encoder query
                # repeat mask of token0 of encoder query n times, where n is the amount of decoder query tokens
                # (mask of example from batch is repeated in example of batch (so, masks of examples are the same))
                encoder_decoder_mask = encoder_decoder_mask.repeat((1, decoder_x_shape[-2], 1))
        else:
            if encoder_padding_mask.shape[-1] >= decoder_x_shape[-2]:
                encoder_decoder_mask = encoder_padding_mask[:decoder_x_shape[-2], :]
            else:
                encoder_decoder_mask = encoder_padding_mask[0,:]  # mask for token with index 0 in encoder query
                # repeat mask of token0 of encoder query n times, where n is the amount of decoder query tokens
                encoder_decoder_mask = encoder_decoder_mask.repeat((decoder_x_shape[-2], 1))

        return encoder_decoder_mask

    def __init__(self, layers_amount, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype):
        super().__init__()
        self._decoder_layer_list = nn.ModuleList()
        self._look_ahead_mask = self._init_look_ahead_mask(max_len, device, dtype)

        for _ in range(layers_amount):
            self._decoder_layer_list.append(
                DecoderLayer(d_model, attention_d_head, ffn_d_hidden, dropout_prob, device, dtype)
            )

    def _count_result_look_ahead_mask(self, padding_mask, x_query_shape):
        # x_query_shape[-2] is amount of query tokens (and key tokens, because it's self-attention)
        mask = self._look_ahead_mask[:x_query_shape[-2], :x_query_shape[-2]]
        if padding_mask is not None:
            mask = mask + padding_mask
        return mask

    def forward(self, decoder_x, encoder_output, encoder_padding_mask=None, decoder_padding_mask=None):
        self_attention_mask = self._count_result_look_ahead_mask(decoder_padding_mask, decoder_x.shape)
        if encoder_padding_mask is not None:
            encoder_decoder_mask = self._count_encoder_decoder_mask(encoder_padding_mask, decoder_x.shape)
        else:
            encoder_decoder_mask = encoder_padding_mask


        for decoder_layer in self._decoder_layer_list:
            decoder_x = decoder_layer(
                decoder_x, encoder_output,
                self_attention_mask=self_attention_mask, encoder_decoder_mask=encoder_decoder_mask
            )
        return decoder_x



