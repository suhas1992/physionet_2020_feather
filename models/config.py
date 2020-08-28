import csv
import torch

# DEFINE PROGRAM CONSTANTS HERE
EPOCH = 50
BATCH_SIZE = 2
RANDOM_SEED = 42
NUM_WORKERS = 6
PIN_MEMORY = True
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.2
TEST_SPLIT = 0.05
CNN = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DICT_PATH = "/home/ubuntu/physionet/Training_WFDB/"
DATA_PATH = "/home/ubuntu/physionet/Training_WFDB/"
OBS_DICT_PATH = "/home/vsanil/workhorse3/physionet/observations.p"
MAPPING_CSV_PATH = "dx_mapping_scored.csv"
DATA_PATH = ""
TARGETS = {}
with open(MAPPING_CSV_PATH) as c:
    reads = csv.reader(c, delimiter=',')
    for row in reads:
        TARGETS[row[1]] = row[2]
TARGET_NAMES = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB','STD','STE']