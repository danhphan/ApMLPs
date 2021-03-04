import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param embedding_matrix: numpy array with vectors for all words
        """
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        # Define an input embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        # Embedding matrix is used as weights of the embedding layer
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # Keep these pretrained embedding intact
        self.embedding.weight.requires_grad = False

        # A simple bidirectional LSTM with hidden size of 128
        self.lstm = nn.LSTM(embed_dim,128, bidirectional=True, batch_first=True,)
        # An output layer using linear
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        # apply mean anx max pooling
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        # Concat and pass though the linear output layer
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.out(out)
        return out
