import torch.nn as nn

import config


class Encoder(nn.Module):

    def __init__(self, vocab_size, padding_idx):

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )

    def forward(self, x):
        
        embeded = self.embedding(x)

        output, h_n = self.gru(embeded)
