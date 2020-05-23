import os
import config as cfg 

import torch
import torch.nn as nn

def train(Model, Trainloader, Optimizer, Criterion, Epoch):
    """ Defines a training structure that considers the model, optimizer 
        and criterion and robtains data from a dataloader one batch at a 
        time.

        Args:
            Model: PyTorch Model
            Trainloader: Iterable dataloader
            Optimizer: Optimizer to update model's parameters
            Criterion: Loss function to compute loss and backpropogate
            Epoch: Training epoch #

        Returns:
            (Model, Optimizer) : Returns the model and optimizer after one training cycle
        """
    for batch_num, (features, labels) in enumerate(Trainloader):
        # Send data to device
        features, labels = features.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        Optimizer.zero_grad()

        pred = Model(features)

        # Compute loss and update optimizer
        loss = Criterion(pred, labels)
        loss.backward()
        Optimizer.step()

        if batch_num % 25 == 1:
            curr_loss = float(loss.item())
            print("Epoch: ", Epoch, "Training Loss: ", curr_loss)

        # Clear redundant variables
        torch.cuda.empty_cache()
        del features
        del labels

    return Model, Optimizer