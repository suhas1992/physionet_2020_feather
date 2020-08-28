import os
import csv  
import psutil
import random
import numpy as np
from scipy.io import loadmat

CLASS_DICT = {1:["270492004", "164889003", "164890007", "426627000", "10370003", "427393009", "426177001", "426783006", "427084000"],
              2:["713427006", "713426002", "164909002", "59118001", "445118002"],
              3:["251146004"],
              4:["284470004","427172004","63593006","17338001"],
              5:["164947007","111975006","164917005","164934002","59931005"],
              6:["39732003","47665007"],
              7:["698252002"],
              8:["427172004","63593006","17338001"]}

RESAMP = 0.25

def extract_individual_group_data(files, snomed_code, diagnosis, group_num):
    """
        Return a feature set for an individual diagnosis 
    """
    # Extract all positive data samples
    feature_dict = {'features':[], 'labels':[]}
    p_count = 0
    count = 0
    for idx, f in enumerate(files):
        if f.endswith('.mat'):
            #print(count, f, idx)
            if count % 10000 == 0:
                print("RAM used: ", psutil.virtual_memory().percent, "Files done: ", count)
            data, header = load_challenge_data(f)
            label = header[-4].replace("#Dx: ","").replace("\n","").split(',')
            add = False 
            for lbl in label:
                if lbl == snomed_code:
                    l = [1,0]
                    add = True
                    p_count += 1
            if add:
                feature_dict['features'].append(data)
                feature_dict['labels'].append(l) 
            count += 1

    # Extract all negative data samples
    in_group, out_group = p_count-int(p_count*RESAMP), int(p_count*RESAMP)
    print(p_count, in_group, out_group)
    random.shuffle(files)
    count = 0
    for idx, f in enumerate(files):
        if in_group == 0 and out_group == 0:
            break
        if f.endswith('.mat'):
            if count % 10000 == 0:
                print("RAM used: ", psutil.virtual_memory().percent, "Files done: ", count)
            data, header = load_challenge_data(f)
            label = header[-4].replace("#Dx: ","").replace("\n","").split(',')
            add = False
            if snomed_code not in label:
                for lbl in label:
                    if lbl in CLASS_DICT[group_num]:
                        if in_group != 0:
                            in_group -= 1
                            l = [0,1]
                            add = True
                            break
                    else:
                        if out_group != 0:
                            out_group -= 1
                            l = [0,1]
                            add = True
                            break
            if add:
                feature_dict['features'].append(data)
                feature_dict['labels'].append(l) 
            count += 1
    
    print("Features for diagnosis {} extracted".format(diagnosis))
    return feature_dict

def get_class_group(diagnoses):
    """
        Return the SNOMED code and the group number of the input 
        diagnoses
    """
    found = False
    snomed_code = ""
    group_num = "200"
    group_elems = []

    with open("dx_mapping_scored.csv") as c:
        reads = csv.reader(c, delimiter=",")
        for row in reads:
            if row[2] == diagnoses:
                found = True
                snomed_code = row[1]
                break

    if not found:
        print("Incorrect diagnoses entered!")
        exit() 

    for key, value in CLASS_DICT.items():
        if snomed_code in value:
            group_num = str(key)
            with open("dx_mapping_scored.csv") as c:
                reads = csv.reader(c, delimiter=",")
                for row in reads:
                    if row[1] in value:
                        group_elems.append(row[2])
            
            break

    return snomed_code, group_num, group_elems

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data

# Extract challenge data for a particular group
def extract_challenge_data(files, group, feature_dict):
    invalid_count = 0
    count = 0

    feature_dict = {'features':[], 'labels':[]}
    labels = [0 for _ in CLASS_DICT[group]]
    keys = {k:idx for idx, k in enumerate(CLASS_DICT[group])}

    for idx, f in enumerate(files):
        if f.endswith('.mat'):
            #print(count, f, idx)
            if count % 1000 == 0:
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

    print("Features for group {} extracted".format(group))
    return feature_dict