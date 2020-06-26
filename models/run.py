import os
import main as mn 
import config as cfg 
from networks.rnn import RNN 
from data import get_loader

import torch
import torch.nn as nn

if __name__ == "__main__":
    # Define model parameters
    train_loader = get_loader("train")
    val_loader = get_loader("val")

    input_dim = 12
    hidden_list = [24, 16, 16, 12, 9]
    output_dim = 9

    #model = MLP(input_dim, hidden_list, output_dim)
    model = RNN(input_dim, input_dim, output_dim)
    model.to(cfg.DEVICE)
    criterion = nn.BCELoss()
    criterion.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.01)

    for i in range(cfg.EPOCH):
        print("\nEpoch number: ", i)
        # Train the model
        model.train()
        model, optimizer = mn.train(model, train_loader, optimizer, criterion, i)

        # Evaluate the model
        model.eval()
        accuracy, loss = mn.eval(model, val_loader, criterion, i)
        
        # Step through the scheduler
        scheduler.step(loss)