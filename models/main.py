import os
import numpy as np
import utils.eval_metrics as em
import config as cfg 

import torch
import pandas as pd 
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix

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
    for batch_num, (features, labels, lengths) in enumerate(Trainloader):
        # Send data to device
        features, labels = features.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        Optimizer.zero_grad()

        pred = Model(features, lengths)

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
        del loss

    return Model, Optimizer

def eval(Model, Evalloader, Criterion, Epoch):
    """ Defines a evaluation structure that considers the model, optimizer 
        and criterion and obtains data from a dataloader one batch at a 
        time.

        Args:
            Model: PyTorch Model
            Evalloader: Iterable dataloader
            Optimizer: Optimizer to update model's parameters
            Criterion: Loss function to compute loss and backpropogate
            Epoch: Training epoch #

        Returns:
            (Accuracy) : Returns the accuracy
        """
    accuracy = 0
    tot_loss = 0
    true_labels = []
    preds = []
    for batch_num, (features, labels, lengths) in enumerate(Evalloader):
        # Send data to device
        features, labels = features.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        pred = Model(features, lengths)

        # Compute loss 
        loss = Criterion(pred, labels)

        pred = torch.where(pred[0,:].cpu() > 0.5, torch.tensor([1.0]),torch.tensor([0.0]))
        labels = labels[0,:]

        pred = pred.detach().numpy()
        labels = labels.cpu().detach().numpy()

        preds.append(pred.tolist())
        true_labels.append(labels.tolist())

        tot_loss += float(loss.item())
        
        if batch_num % 50 == 1:
            curr_loss = float(loss.item())
            print("Epoch: ", Epoch, "Validation Loss: ", curr_loss)

        # Clear redundant variables
        torch.cuda.empty_cache()
        del features
        del labels
        del loss

    # Compute final metrics
    true_labels = np.vstack(true_labels)
    preds = np.vstack(preds)

    accuracy, precision, recall, misclass_rate = em.print_multilabel_report(true_labels, preds)

    print("\n\n\nTotal Accuracy: ", accuracy, 
          "Total Misclassification Rate: ", misclass_rate,
          "Total Recall: ", recall, 
          "Total Precision: ", precision)
    
    return accuracy, tot_loss/(batch_num+1)