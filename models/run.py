import os
import math 
import argparse 
import main as mn 
import config as cfg 
import numpy as np
from scipy.io import loadmat
from networks.rnn import RNN 
from networks.mobile import mobileNet
from networks.resnext import ResNet, Bottleneck, BasicBlock
from torchsummary import summary 
import data as dt
#from data import get_loader

import torch
import torch.nn as nn

def save(model,optimizer,path):
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
               },os.path.join(path,"best_model.pth"))

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data

def extract_challenge_data(files):
    invalid_count = 0
    count = 0
    feature_dict = {'features':[], 'labels':[]}
    labels = [0 for _ in list(cfg.TARGETS.keys())[1:]]
    keys = {k:idx for idx, k in enumerate(list(cfg.TARGETS.keys())[1:])}
    print(keys)

    for idx, f in enumerate(files):
        if f.endswith('.mat'):
            print(count, f, idx)
            data, header = load_challenge_data(f)
            label = header[-4].replace("#Dx: ","").replace("\n","").split(',')
            l = labels
            for lbl in label:
                l[keys[lbl]] = 1
            feature_dict['features'].append(data)
            feature_dict['labels'].append(l) 
            count += 1
            print("Here", count)

    return feature_dict, invalid_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--datadir", required=True,
                        help="Complete input directory")
    args = parser.parse_args()

    if not os.path.exists(args.datadir):
        print("Path does not exist")
        exit()

    cfg.DATA_PATH = args.datadir
    files = [os.path.join(cfg.DATA_PATH, f) for f in os.listdir(cfg.DATA_PATH)]
    feature_dict, invalid_count = extract_challenge_data(files)

    print("Number of invalid files: ", invalid_count)
    dt.FEATURE_DICT = feature_dict
    
    # Define model parameters
    train_loader = dt.get_loader("train")
    val_loader = dt.get_loader("val")

    input_dim = 12
    hidden_list = [24, 16, 16, 12, 9]

    # Model params for mobilenet
    mobile_model_params = [
    # t   c   n  s
    [ 1,  16, 1, 1],
    [ 6,  24, 3, 1],
    [ 6,  32, 4, 2],
    [ 6,  64, 5, 3],
    [ 6,  96, 3, 2],
    [ 6, 160, 3, 2],
    [ 6, 320, 1, 1]]

    output_dim = len(cfg.TARGETS)-1

    #model = MLP(input_dim, hidden_list, output_dim)
    #model = RNN(input_dim, input_dim, output_dim)
    #model = mobileNet(input_dim, mobile_model_params, output_dim)
    #"""
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4)
    #"""
    model.to(cfg.DEVICE)

    # Print model summary
    summary(model, (12, 10000))

    # Define training parameters
    path = "/share/workhorse3/vsanil/physionet/best_models/"
    criterion = nn.BCELoss()
    criterion.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.01, mode='max')
    best_f1 = 0.0

    for i in range(cfg.EPOCH):
        print("\nEpoch number: ", i)
        # Train the model
        model.train()
        model, optimizer = mn.train(model, train_loader, optimizer, criterion, i)

        # Evaluate the model
        model.eval()
        accuracy, loss, recall, precision = mn.eval(model, val_loader, criterion, i)

        # Compute f1 and check if it is nan
        f1 = 2*((precision * recall) / (precision + recall))
        f1 = float(f1)
        if math.isnan(f1):
            f1 = 0.0
        
        if f1 > best_f1:
            save(model, optimizer, path)
            best_f1 = f1

        # Step through the scheduler
        scheduler.step(accuracy)