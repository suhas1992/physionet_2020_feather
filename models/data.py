import os 
import pickle 
import config as cfg 

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader

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
    features, labels = zip(*batch)

    features = rnn.pad_sequence(features, batch_first=True)
    labels = torch.cat(list(labels))

    return features, labels

# Dictionary to define Dataset stats
config_dict = {'train':{'dataset':ECGTrainSet, 'collate':train_collate, 'shuffle':True}}

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
    dataset = config_dict[loader_type]['dataset'](feature_dict)

    loader = DataLoader(dataset, 
                        num_workers=cfg.NUM_WORKERS, 
                        pin_memory=cfg.PIN_MEMORY, 
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=config_dict[loader_type]['shuffle'],
                        collate_fn=config_dict[loader_type]['collate'])

    return loader