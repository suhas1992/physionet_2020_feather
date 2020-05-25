import os 
import pickle 
import numpy as np
import config as cfg 

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, random_split, sampler 

class ECGTrainSet(Dataset):
    def __init__(self, feature_dict):
        self.features = feature_dict['Feature']
        self.label = feature_dict['Label']
        self.strategy = "expand labels"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        f = torch.transpose(torch.from_numpy(self.features[index]).float(), 0, 1)

        l = torch.LongTensor(self.label[index]).unsqueeze(0)

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
    lens = [len(seq) for seq in features]
    labels = torch.cat(list(labels))

    return features, labels, lens

def train_val_test_split(dataset):
    """ Function to create samplers that will be used to
        split the dataset given a certain random seed.

        Args:
            dataset: the full input dataset to be split

        Returns:
            training data sampler, validation data sampler and test data sampler
    """
    dataset_size = len(dataset)
    train_size = int(0.75 * dataset_size)
    val_size = int(0.15 * dataset_size)
    
    # Create indices and shuffle them
    indices = list(range(dataset_size))
    np.random.seed(cfg.RANDOM_SEED)
    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

    # Create samplers
    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)
    test_sampler = sampler.SubsetRandomSampler(test_indices)

    return train_sampler, val_sampler, test_sampler

# Dictionary to define Dataset stats
train_sampler, val_sampler, test_sampler = train_val_test_split(ECGTrainSet)
config_dict = {'train':{'collate':train_collate, 'shuffle':True, 'sampler':train_sampler},
               'val':{'collate':train_collate, 'shuffle':False, 'sampler':val_sampler}}

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

    loader = DataLoader(dataset, 
                        batch_sampler=config_dict[loader_type]['sampler'],
                        num_workers=cfg.NUM_WORKERS, 
                        pin_memory=cfg.PIN_MEMORY, 
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=config_dict[loader_type]['shuffle'],
                        collate_fn=config_dict[loader_type]['collate'])

    return loader