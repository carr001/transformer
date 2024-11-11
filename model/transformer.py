# Author: Vodohleb04
import torch
from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.transformer_embedding import TransformerEmbedding


class Transformer(nn.Module):

    def __init__(
        self,
        encoder_layers_amount, decoder_layers_amount,
        encoder_vocab_size, decoder_vocab_size,
        d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len,
        device, dtype, padding_index=None, encoder_mask=None
    ):
        super().__init__()
        self._encoder_embedding = TransformerEmbedding(
            encoder_vocab_size, d_model, max_len, dropout_prob, device, dtype, padding_index
        )
        self._decoder_embedding = TransformerEmbedding(
            decoder_vocab_size, d_model, max_len, dropout_prob, device, dtype
        )

        self._encoder = Encoder(
            encoder_layers_amount, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype,
            custom_mask=encoder_mask
        )
        self._decoder = Decoder(
            decoder_layers_amount, d_model, attention_d_head, ffn_d_hidden, dropout_prob, max_len, device, dtype,
            encoder_mask=encoder_mask
        )

        self._output_linear = nn.Linear(d_model, decoder_vocab_size, bias=True, device=device, dtype=dtype)
        nn.init.xavier_normal_(self._output_linear.weight)


    def forward(self, input_sequence, output_sequence, return_logits=False):
        encoder_x = self._encoder_embedding(input_sequence)
        decoder_x = self._decoder_embedding(output_sequence)

        encoder_x = self._encoder(encoder_x)
        decoder_x = self._decoder(decoder_x=decoder_x, encoder_output=encoder_x)

        decoder_x = self._output_linear(decoder_x)

        if return_logits:
            return decoder_x
        else:
            return nn.functional.softmax(decoder_x, dim=-1)


if __name__ == "__main__":
    e_x = torch.as_tensor(
        [
            [
                1,
                2,
                3
            ],
            [
                10,
                11,
                12
            ]
        ], device=torch.device("cuda")
    )
    d_x = torch.as_tensor(
        [
            2
        ],
        device=torch.device("cuda")
    )

    transformer = Transformer(
        2, 2, 20, 6, 6, 2, 12, 0.1, 3,
        torch.device("cuda"), torch.float32
    )

    value = transformer(e_x, d_x, return_logits=False)
    print(value)



