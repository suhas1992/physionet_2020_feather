import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_list, output_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_list[0]))
        #layers.append(nn.BatchNorm1d(hidden_list[0]))
        layers.append(nn.ReLU())
        for idx, size in enumerate(hidden_list):
            if idx == len(hidden_list)-1:
                layers.append(nn.Linear(size, output_dim))
                continue
            layers.append(nn.Linear(size, hidden_list[idx+1]))
            #layers.append(nn.BatchNorm1d(hidden_list[idx+1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, lengths):
        #x = rnn.pack_padded_sequence(x)
        return self.mlp(x)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.embedding_size = input_size
        self.bilstm = nn.LSTM(self.embedding_size, 
                              hidden_size, 
                              num_layers=2, 
                              dropout=0.1, 
                              bidirectional=True,
                              batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, output_size*2)
        self.linear2 = nn.Linear(output_size*2, output_size)
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()

    def forward(self, x, lengths):
        packed_x = rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        packed_out, (hidden, cell) = self.bilstm(packed_x)
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        #out, out_lens = rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.linear1(out)
        out = self.dropout(out)

        out = self.linear2(out)

        out = self.sig(out)

        return out