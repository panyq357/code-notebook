import torch
import torch.nn as nn

import config


class ReviewAnalyzeModel(nn.Module):

    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM, padding_idx=padding_index)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

    def forward(self, x):

        embed = self.embedding(x)

        output, (h_n, c_n) = self.lstm(embed)
        # output.shape: [batch_size, seq_len, hidden_size]
        
        last_nonzero_index = (x != self.embedding.padding_idx).sum(dim=1) - 1
        last_hidden = output[torch.arange(x.shape[0], device=x.device), last_nonzero_index, :]
        # last_hidden.shape: [batch_size, hidden_size]

        output = self.linear(last_hidden)
        # output.shape: [batch_size, 1]

        # squeeze to one dimension
        return torch.squeeze(output)
        
