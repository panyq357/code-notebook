import torch.nn as nn
import config


class MyModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x):

        # x.shape: [batchsize, seq_len]

        embed = self.embedding(x)
        # embed.shape: [batchsize, seq_len, embedding_dim]

        output, h_n = self.rnn(embed)
        # output is the hidden state of all time step in last layer. output.shape: [batchsize, seq_len, hidden_size]
        # hn is the the hidden state of last time step in all layer. h_n.shape: [layer_number, batch_size, hidden_size]

        last_hidden_state = h_n[-1,:,:]
        # last_hidden_state.shape: [batch_size, hidden_size]

        output = self.linear(last_hidden_state)
        # ouput.shape: [batch_size, vocab_size]

        return output
