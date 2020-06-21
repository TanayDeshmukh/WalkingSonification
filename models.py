import sys
import torch
import numpy as np
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, num_features, embedding_dim):
        super(Encoder, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=embedding_dim,
                            bidirectional=True)

    def forward(self, sequence):
        x, (hidden_n, cell_n) = self.lstm(sequence)

        return x, (hidden_n, cell_n)

class Decoder(nn.Module):
    
    def __init__(self, embedding_dim, num_features, sequence_length):
        super(Decoder, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim * 2
        self.sequence_length = sequence_length
    
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.num_features,
                            bidirectional=False)

    def forward(self, sequence, hidden_state, cell_state):

        predictions = torch.zeros(self.sequence_length, hidden_state.shape[-2], hidden_state.shape[-1]*hidden_state.shape[0]).to(device)
        for i in range(self.sequence_length):
            x, (hidden_state, cell_state) = self.lstm(sequence, (hidden_state, cell_state))
            predictions[i] = x

        return predictions

class AutoEncoder(nn.Module):

    def __init__(self, num_features, embedding_dim, sequence_length, batch_size, device):
        super(AutoEncoder, self).__init__()

        self.num_features = num_features
        self.batch_size = batch_size
        self.device = device
        self.encoder = Encoder(num_features, embedding_dim)
        self.decoder = Decoder(embedding_dim, num_features, sequence_length)
    
    def init_decoder_hidden_states(self):
        # Change the first dimension to 2 if using bidirectional
        hidden_state = torch.zeros(1, self.batch_size, self.num_features).to(self.device)
        cell_state = torch.zeros(1, self.batch_size, self.num_features).to(self.device)
        return hidden_state, cell_state

    def forward(self, sequences):
        out, (h, c) = self.encoder(sequences)
        h_reshaped = h.view(1, h.shape[1], -1)
        hidden_state, cell_state = self.init_decoder_hidden_states()
        out = self.decoder(h_reshaped, hidden_state, cell_state)
        return out
