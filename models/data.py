import os 
import pickle 
import config as cfg 

import torch
from torch.utils.data import Dataset, DataLoader

class ECGTrainSet(Dataset):
    def __init__(self, feature_dict):
        features = feature_dict['Feature']
        label = feature_dict['Label']

        print(len(features))
        print(len(label))

# Dictionary to define Dataset stats
config_dict = {'train':{'dataset':ECGTrainSet}}

def load_pickle(path):
    """ Load data stored in pickle format

        Args:
            path: data path

        Returns:
            loaded data
    """
    with open(path, 'rb') as fp:
        data = pickle.load(fp)

    return data

def get_loader(loader_type):
    feature_dict = load_pickle(cfg.OBS_DICT_PATH)
    dataset = config_dict[loader_type]['dataset'](feature_dict)