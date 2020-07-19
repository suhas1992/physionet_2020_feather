import os
import sys
sys.path.append("models")
import math 
import pickle 
import psutil
import argparse 
import main as mn 
import csv
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

# Save data in pickle format
def save_pickle(path, data_dict, filename):
    with open(os.path.join(path, '{}.p'.format(filename)), 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data

def extract_challenge_data(files, classes):
    invalid_count = 0
    count = 0

    feature_dict = {'features':[], 'labels':[]}
    labels = [0 for _ in classes]
    keys = {k:idx for idx, k in enumerate(classes)}

    for idx, f in enumerate(files):
        if f.endswith('.mat'):
            #print(count, f, idx)
            if count % 5000 == 0:
                print("RAM used: ", psutil.virtual_memory().percent, "Files done: ", count)
            data, header = load_challenge_data(f)
            label = header[-4].replace("#Dx: ","").replace("\n","").split(',')
            l = labels.copy()
            add = False 
            for lbl in label:
                try:
                    l[keys[lbl]] = 1
                    add = True
                except KeyError:
                    continue
            if add:
                feature_dict['features'].append(data)
                feature_dict['labels'].append(l) 
            count += 1

    return feature_dict, invalid_count

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

def train(input_directory, output_directory, classes, val_exists=True, use_dict=False):
    cfg.DATA_PATH = input_directory
    
    files = [os.path.join(cfg.DATA_PATH, f) for f in os.listdir(cfg.DATA_PATH)]
    feature_dict, invalid_count = extract_challenge_data(files, classes)
    print("Data extracted")

    #dt.FEATURE_DICT = feature_dict
    #del feature_dict
    print(len(feature_dict))
    # Define model parameters
    train_loader = dt.get_loader("train", val_exists, feature_dict)
    if val_exists:
        val_loader = dt.get_loader("val", val_exists, feature_dict)

    output_dim = len(classes)
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4)
    model.to(cfg.DEVICE)

    # Define training parameters
    path = output_directory
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
        accuracy, loss, recall, precision = mn.eval(model, val_loader, criterion, i, classes=classes)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--datadir", required=True,
                        help="Complete input directory")
    parser.add_argument("-c","--check", required=False, default="False",
                        help="Check best model's performance")  
    args = parser.parse_args()

    if not os.path.exists(args.datadir):
        print("Path does not exist")
        exit()

    
    cfg.DATA_PATH = args.datadir

    # Check if observation dictionary already exists, don't extract if yes
    if os.path.exists(os.path.join(args.datadir, 'obs.p')):
        print("Data exists, load features")
        with open(os.path.join(args.datadir, 'obs.p'), 'rb') as fp:
            feature_dict = pickle.load(fp)
    else:
        files = [os.path.join(cfg.DATA_PATH, f) for f in os.listdir(cfg.DATA_PATH)]
        feature_dict, invalid_count = extract_challenge_data(files)
        print("Data extracted")
        save_pickle(args.datadir, feature_dict, 'obs')

    # Define model parameters
    train_loader = dt.get_loader("train", feature_dict=feature_dict)
    val_loader = dt.get_loader("val", feature_dict=feature_dict)

    input_dim = 12
    hidden_list = [24, 16, 16, 12, 9]

    # Model params for mobilenet
    mobile_model_params = [
    # t   c   n  s
    [ 1,  16, 1, 1],
    [ 6,  24, 3, 1],
    [ 6,  32, 4, 2],
    [ 6,  64, 5, 2],
    [ 6,  96, 3, 2],
    [ 6, 160, 3, 1],
    [ 6, 320, 1, 1]]

    output_dim = len(cfg.TARGETS)-1

    #model = MLP(input_dim, hidden_list, output_dim)
    #model = RNN(input_dim, input_dim, output_dim)
    #model = mobileNet(input_dim, mobile_model_params, output_dim)
    #"""
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4,
                   all_depthwise=True, 
                   )
    #"""
    model.to(cfg.DEVICE)

    # Define training parameters
    path = "/share/workhorse3/vsanil/physionet/best_models/"
    criterion = nn.BCELoss()
    criterion.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.01, mode='max')
    best_f1 = 0.0

    if args.check == "True":
        print("Checking best model's perfomance")
        best_model_path = "/home/vsanil/workhorse3/physionet/best_models/best_model.pth"

        # Load the model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with open('checkfile.log', 'w') as f:
            mn.eval(model, val_loader, criterion, 0, f)
        print("Best model stats printed to checkfile.log")
        exit()

    # Print model summary
    summary(model, (12, 10000))
    num_classes = get_classes(args.datadir)

    with open("checkfile.log", "w") as f:
        print("::::::::: Logs ::::::::::", file=f)

    for i in range(cfg.EPOCH):
        print("\nEpoch number: ", i)
        # Train the model
        model.train()
        model, optimizer = mn.train(model, train_loader, optimizer, criterion, i)

        # Evaluate the model
        model.eval()
        with open("checkfile.log", "a") as f:
            accuracy, loss, recall, precision = mn.eval(model, val_loader, criterion, i, f, classes=num_classes)

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