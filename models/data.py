import os 
import pickle 
import numpy as np
import config as cfg 
from sklearn.model_selection import train_test_split 

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, random_split, sampler, Subset

class ECGTrainSet(Dataset):
    def __init__(self, feature_dict):
        self.features = feature_dict['Feature']
        self.label = feature_dict['Label']
        self.strategy = "expand labels"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        f = torch.transpose(torch.from_numpy(self.features[index]).float(), 0, 1)

        l = torch.FloatTensor(self.label[index]).unsqueeze(0)

        return f, l

def train_collate(batch):
    """ Function to collate a single batch and generate padding across 
        features along with lengths of the original features before
        padding.

        Args:
            batch: a single batch prepared from dataloader

        Returns:
            the batch split across padded features, labels and lenghts of each feature
    """
    features, labels = zip(*batch)

    features = rnn.pad_sequence(features, batch_first=True)
    if cfg.CNN:
        features = torch.transpose(features, 1, 2)
    lens = [len(seq) for seq in features]
    labels = torch.cat(list(labels))

    return features, labels, lens

# Dictionary to define Dataset stats
config_dict = {'train':{'collate':train_collate, 'shuffle':True, 'batch_size':cfg.BATCH_SIZE},
               'val':{'collate':train_collate, 'shuffle':False, 'batch_size':1}}

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
    """ Returns a PyTorch dataloader object for computation
        that batches features according to a collate function

        Args:
            loader_type : type of dataloader required (train/val/test)

        Returns:
            dataloader object
    """
    feature_dict = load_pickle(cfg.OBS_DICT_PATH)
    dataset = ECGTrainSet(feature_dict)
    #train_sampler, val_sampler, test_sampler = train_val_test_split(dataset)
    #config_dict['train']['sampler'] = train_sampler
    #config_dict['val']['sampler'] = val_sampler
    loader_set = None
    if loader_type == "train":
        train_idx, _ = train_test_split(list(range(len(dataset))), 
                                        test_size=cfg.VAL_SPLIT+cfg.TEST_SPLIT,
                                        shuffle=True, random_state=cfg.RANDOM_SEED)
        loader_set = Subset(dataset, train_idx)
    elif loader_type == "val" or loader_type == "test":
        _, temp_idx = train_test_split(list(range(len(dataset))), 
                                       test_size=cfg.VAL_SPLIT+cfg.TEST_SPLIT,
                                       shuffle=True, random_state=cfg.RANDOM_SEED)
        temp_set = Subset(dataset, temp_idx)
        val_idx, test_idx = train_test_split(list(range(len(temp_set))),
                                             test_size= 1-(cfg.TEST_SPLIT/(cfg.VAL_SPLIT + cfg.TEST_SPLIT)),
                                             shuffle=True, random_state=cfg.RANDOM_SEED)
        if loader_type == "val":
            loader_set = Subset(temp_set, val_idx)
        else:
            loader_set = Subset(temp_set, test_idx) 

    loader = DataLoader(loader_set, 
                        num_workers=cfg.NUM_WORKERS, 
                        pin_memory=cfg.PIN_MEMORY, 
                        batch_size=config_dict[loader_type]['batch_size'],
                        shuffle=config_dict[loader_type]['shuffle'],
                        collate_fn=config_dict[loader_type]['collate'])

    return loader
