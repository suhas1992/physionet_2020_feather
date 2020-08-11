import os
import sys
sys.path.append("models")

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

def load_model(model, path):
    """
        Return the loaded model
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data

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

def save_plots(index, sig, path, preds, group_num):
    
    pred_dict = {group_num[i]:preds[i] for i in range(len(group_num))}

    fig, ax = plt.subplots(nrows=len(sig), ncols=1, figsize=(20,40))
    fig.suptitle("Preds: {}".format(pred_dict))
    count = 0

    for idx, row in enumerate(ax):
        row.plot(sig[count])
        ax[idx].set_title("Lead {}".format(count))
        count += 1

    fig.savefig(os.path.join(path,"{}.png".format(index)))
    plt.close(fig)

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
                              'best_model_{}.pth'.format(group_num))

    if group_num == 0:
        output_dim = len(cfg.TARGETS)-1
        num_classes = get_classes(args.datadir)
    else:
        output_dim = len(cdict.CLASS_DICT[group_num])
        num_classes = cdict.CLASS_DICT[group_num]

    # Model params
    print(output_dim)
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4,
                   all_depthwise=True, 
                   )
    model.to(cfg.DEVICE)

    if not os.path.exists(model_path):
        # Train the model again
        cmd = "python models/run.py -i {0} -g {1}".format(args.datadir, group_num)
    else:
        model = load_model(model, model_path)

    plot_path = "plots/"
    window_len = 2000
    count = 10
    
    files = [os.path.join(args.datadir, f) for f in os.listdir(args.datadir)]
    diag_sigs = []

    # Check for files and seach for file with snomed code
    for f in files:
        if count == 0:
            break

        if f.endswith('.mat'):
            data, header = load_challenge_data(f)
            label = header[-4].replace("#Dx: ","").replace("\n","").split(',')
            if snomed in label:
                diag_sigs.append(data)
                count -= 1

    # Window the signal and check the window for dominant diagnosis
    for idx, sigs in enumerate(diag_sigs):
        sigpath = os.path.join(plot_path, str(idx))
        count = 0
        if os.path.exists(sigpath):
            # Clean the directory
            shutil.rmtree(sigpath)
            os.mkdir(sigpath)
        else:
            os.mkdir(sigpath)
        for i in range(0, sigs.shape[1], window_len):
            count += 1
            sig = sigs[:,i:i+window_len]
            inp = torch.from_numpy(sig).float().unsqueeze(0).to(cfg.DEVICE)
            preds = model(inp)[0].cpu().detach().numpy()
            save_plots(count, sig, sigpath, preds, group_ls)