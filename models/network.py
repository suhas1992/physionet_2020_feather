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

    def forward(self, x):
        #x = rnn.pack_padded_sequence(x)
        return self.mlp(x)