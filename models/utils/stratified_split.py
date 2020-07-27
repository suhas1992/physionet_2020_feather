import numpy as np
import random

def train_test_split(idxs, dataset, test_size, shuffle, random_state):
    count_dict = {}
    data_len = len(dataset)
    percent = test_size
    for i in idxs:
        tup_l = repr(dataset[i][1].numpy().tolist())
        if tup_l not in count_dict:
            count_dict[tup_l] = {"idx":[i],
                                 "count":1}
        else:
            count_dict[tup_l]["idx"].append(i)
            count_dict[tup_l]["count"] += 1

    train_idxs = []
    test_idxs = []
    for key, values in count_dict.items():
        train_split = values["count"] - int(values["count"]*percent)
        train_idxs.extend(values["idx"][:train_split])
        test_idxs.extend(values["idx"][train_split:])
    
    if shuffle:
        random.seed(random_state)
        random.shuffle(train_idxs)
        random.shuffle(test_idxs)

    return train_idxs, test_idxs
