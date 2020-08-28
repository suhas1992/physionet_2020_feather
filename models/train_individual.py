import os
import sys

import csv
import math 
import pickle 
import argparse 
import shutil
import main as mn 
import data as dt
import config as cfg 
import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt 
import utils.get_class_dict as cdict

import torch
import torch.nn as nn
from torchsummary import summary 
from networks.mobile import mobileNet
from networks.resnext import ResNet, Bottleneck, BasicBlock

def save(model,optimizer,path,group):
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
               },os.path.join(path,"best_model_{}.pth".format(group)))

# Save data in pickle format
def save_pickle(path, data_dict, filename):
    with open(os.path.join(path, '{}.p'.format(filename)), 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Find unique classes.
def get_classes(input_directory, train_27=False):
    classes = set()
    check_27_class = set()
    with open("dx_mapping_scored.csv") as c:
        reads = csv.reader(c, delimiter=',')
        for row in reads:
            check_27_class.add(row[1])

    for f in os.listdir(input_directory):
        filename = os.path.join(input_directory, f)
        if filename.endswith('.hea'):
            with open(filename, 'r') as f:
                for l in f:
                    if l.startswith('#Dx'):
                        tmp = l.split(': ')[1].split(',')
                        for c in tmp:
                            if c in check_27_class:
                                classes.add(c.strip())

    return sorted(classes)

if __name__=="__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument("-i", "--datadir", required=True,
                        help="Input directory")
    parser.add_argument("-e", "--examples", required=False, default="10",
                        help="Number of examples")
    parser.add_argument("-g","--group", required=False, default="None",
                        help="Group number(0-7)")
    parser.add_argument("-d", "--diagnosis", required=True,
                        help="Refer to Dx mapping csv to get a list of diagnosis")
    args = parser.parse_args() 

    if not os.path.exists(args.datadir):
        print("Path does not exist")
        exit()

    snomed, group_num, group_ls = cdict.get_class_group(args.diagnosis)

    if args.group != "None":
        group_num = int(args.group)
    else:
        group_num = int(group_num)

    model_path = os.path.join('/'.join(args.datadir.split('/')[:-2]),
                              'best_models',
                              'diagnosis_models',
                              'best_model_{}.pth'.format(args.diagnosis))

    # Load the data
    files = [os.path.join(cfg.DATA_DICT_PATH, f) for f in os.listdir(cfg.DATA_DICT_PATH)]
    feature_dict = cdict.extract_individual_group_data(files, snomed, args.diagnosis, group_num)

    # Define model parameters
    train_loader = dt.get_loader("train", feature_dict=feature_dict)
    val_loader = dt.get_loader("val", feature_dict=feature_dict)
    num_classes = get_classes(args.datadir)

    # Binary classification
    output_dim = 2
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4,
                   all_depthwise=True, 
                   )
    model.to(cfg.DEVICE)

    # Training parameters
    model_path = "/home/ubuntu/physionet/best_models/diagnosis_models"
    log_path = "/home/ubuntu/physionet/physionet_2020_feather/diagnosis_logs"
    criterion = nn.BCELoss()
    criterion.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=0.01, mode='max')
    best_f1 = 0.0

    # Print model summary
    summary(model, (12, 10000))

    # Initialize log file
    with open(os.path.join(log_path, "{}.log".format(args.diagnosis)), "w") as f:
        print("::::::::: Logs ::::::::::", file=f)

    for i in range(cfg.EPOCH):
        print("\nEpoch number: ", i)
        # Train the model
        model.train()
        model, optimizer = mn.train(model, train_loader, optimizer, criterion, i)

        # Evaluate the model
        model.eval()
        with open(os.path.join(log_path, "{}.log".format(args.diagnosis)), "a") as f:
            accuracy, loss, recall, precision = mn.ind_eval(model, val_loader, criterion, i, args.diagnosis, f)

        # Compute f1 and check if it is nan
        f1 = 2*((precision * recall) / (precision + recall))
        f1 = float(f1)
        if math.isnan(f1):
            f1 = 0.0
        
        if f1 > best_f1:
            save(model, optimizer, model_path, args.diagnosis)
            best_f1 = f1

        # Step through the scheduler
        scheduler.step(accuracy)