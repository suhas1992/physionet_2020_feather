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

    input_dim = 12
    hidden_list = [24, 16, 16, 12, 10, 9]
    output_dim = 9

    model = MLP(input_dim, hidden_list, output_dim).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01)

    for i in range(cfg.EPOCH):
        print("Epoch number: ", i)
        model.train()
        model, optimizer = mn.train(model, train_loader, optimizer, criterion, i)