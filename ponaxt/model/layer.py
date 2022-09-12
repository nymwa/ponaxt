import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class PonaXTLayer(nn.Module):

    def __init__(self, d_model, d_rnn, dropout, nonlinearity = 'relu'):
        super().__init__()
        self.d_model = d_model
        self.d_rnn = d_rnn
        self.fc1 = nn.Linear(d_model, d_rnn)
        self.fc2 = nn.Linear(d_rnn, d_model)
        self.rnn = nn.RNN(d_rnn, d_rnn, nonlinearity = nonlinearity, bidirectional = True)
        self.act = nn.ReLU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward_rnn(self, x, lengths, h, enforce_sorted):
        packed = pack(x, lengths, enforce_sorted = enforce_sorted)
        output, h = self.rnn(packed, h)
        x, _ = unpack(output)
        x = x[:, :, :self.d_rnn] + x[:, :, -self.d_rnn:]
        return x, h

    def forward_main(self, x, lengths, h, enforce_sorted):
        x = self.fc1(x)
        x = self.act(x)
        x, h = self.forward_rnn(x, lengths, h, enforce_sorted)
        x = self.fc2(x)
        return x, h

    def forward(self, x, lengths, h = None, enforce_sorted = True):
        z, h = self.forward_main(x, lengths, h, enforce_sorted)
        z = self.dropout(z)
        x = self.ln(x + z)
        return x, h

