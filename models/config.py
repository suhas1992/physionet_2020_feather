import torch

# DEFINE PROGRAM CONSTANTS HERE
EPOCH = 10
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 6
PIN_MEMORY = True
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.2
TEST_SPLIT = 0.05
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/vsanil/workhorse3/physionet/Training_WFDB/"
DATA_DICT_PATH = "/home/vsanil/workhorse3/physionet/data.p"
OBS_DICT_PATH = "/home/vsanil/workhorse3/physionet/observations.p"
TARGET_NAMES = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB','STD','STE']