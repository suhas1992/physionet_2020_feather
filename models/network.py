import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class MLP(nn.Module):
    def __init__(self, intput_dim, hidden_list, output_dim):
        super(MLP, self).__init__()
        layers = []

        for idx, size in enumerate(hidden_list):
            if idx==0:
                layers.append(nn.Linear(intput_dim, size))
                layers.append(nn.ReLU())
                continue
            if idx == len(hidden_list)-1:
                layers.append(nn.Linear(size, output_dim))
                layers.append(nn.ReLU())
                continue
            layers.append(nn.Linear(size, hidden_list[idx+1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = rnn.pack_padded_sequence(x)
        return self.mlp(x)