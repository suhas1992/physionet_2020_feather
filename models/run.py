import os
import main as mn 
import config as cfg 
from data import get_loader

if __name__ == "__main__":
    # Define model parameters
    train_loader = get_loader("train")

    for i in range(cfg.EPOCH):
        print(i)