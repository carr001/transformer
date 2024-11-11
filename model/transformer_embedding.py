# Author: Vodohleb04
import torch
from torch import nn
from model.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, dropout_prob, device, dtype, padding_index=None):
        super().__init__()
        self._dropout_prob = dropout_prob

        self._token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_index, device=device, dtype=dtype
        )
        self._positional_encoding = PositionalEncoding(d_model, max_len, device, dtype)


    def forward(self, x):
        token_embedding = self._token_embedding(x)
        positional_encoding = self._positional_encoding(x)
        x = token_embedding + positional_encoding
        x = nn.functional.dropout(x, p=self._dropout_prob, training=self.training)
        return x


