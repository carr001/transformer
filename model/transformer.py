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
        self._device = device
        self._max_len = max_len


    def forward(self, input_sequence, output_sequence):
        encoder_x = self._encoder_embedding(input_sequence)
        decoder_x = self._decoder_embedding(output_sequence)

        encoder_x = self._encoder(encoder_x)
        decoder_x = self._decoder(decoder_x=decoder_x, encoder_output=encoder_x)

        decoder_x = self._output_linear(decoder_x)  # Outputs (batch_size, seq_len, vocab_size).
        # model takes output_sequence in form (<BOS>, token1, token2, ..., last_token) (no <eos> is passed)
        # in cross-entropy <BOS> should predict token1, token1 -> token2, ..., last_token -> <eos>
        # In cross-entropy target output has form (token1, token2, ..., last_token, <EOS>)
        # Model output in cross-entropy - (<BOS>_prediction, token1_prediction, token2_prediction, ..., last_token_prediction)
        # In the best case model output is (token1, token2, ..., last_token, <EOS>)
        # Cross-entropy requires reshape (it takes (batch_size, num_classes) shape. Every token is part of such batch)
        # softmax is taken across -1 dim (for every token of sequence).
        return decoder_x


    def generate(self, input_sequence, bos_token_id, eos_token_id):
        # Takes one-example input (not a batch)
        output_sequence = torch.as_tensor([bos_token_id], device=self._device)
        encoder_x = self._encoder_embedding(input_sequence)
        encoder_x = self._encoder(encoder_x)

        for _ in range(self._max_len):
            decoder_x = self._decoder_embedding(output_sequence)
            decoder_output = self._decoder(decoder_x=decoder_x, encoder_output=encoder_x)
            # only the last token is used for prediction (it knows the context of other tokens)
            decoder_output = decoder_output[-1, :]

            decoder_output = self._output_linear(decoder_output)

            decoder_output = nn.functional.softmax(decoder_output, dim=-1)  # softmax is taken across -1 dim (for every token)
            _, decoder_output = torch.max(decoder_output, dim=-1)  # the id of output token

            output_sequence = torch.concat((output_sequence, decoder_output.unsqueeze(dim=0)))

            if decoder_output.item() == eos_token_id:
                break
        return output_sequence







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
            [2,
             4,
             1],
            [2,
             3,
             1]
        ],
        device=torch.device("cuda")
    )

    transformer = Transformer(
        2, 2, 20, 7, 6, 2, 12, 0.1, 30,
        torch.device("cuda"), torch.float32
    )

    value = nn.functional.softmax(transformer(e_x, d_x), dim=-1)

    input_tokens = torch.as_tensor([3, 5, 2], device = torch.device("cuda"))
    print(value)
    print("generating...")
    with torch.no_grad():
        print(transformer.generate(input_tokens, 5, 6))




