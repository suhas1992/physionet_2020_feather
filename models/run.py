import os
import main as mn 
import config as cfg 
from network import MLP
from data import get_loader

import torch
import torch.nn as nn

if __name__ == "__main__":
    # Define model parameters
    train_loader = get_loader("train")

    input_dim = 2048
    hidden_list = [1024,512,256,128,64,16]
    output_dim = 9

    model = MLP(input_dim, hidden_list, output_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(cfg.EPOCH):
        print("Epoch number: ", i)
        model, optimizer = mn.train(model, train_loader, optimizer, criterion, i)