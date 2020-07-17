import numpy as np, os, sys
from scipy.io import loadmat
from models.run import train
from get_12ECG_features import get_12ECG_features

def train_12ECG_classifier(input_directory, output_directory):
    # Write to checks.txt : USE_DRIVE, USE_VAL, NUM_CLASSES  
    config_vals = ["True", "True", 27]
    with open('checks.txt', 'w') as f:
        for item in config_vals:
            f.write("{}\n".format(str(item)))
    
    # Get classes and train model
    classes = get_classes(input_directory)
    train(input_directory, output_directory, classes)

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes(input_directory):
    classes = set()
    for filename in input_directory:
        if filename.endswith('.hea'):
            with open(filename, 'r') as f:
                for l in f:
                    if l.startswith('#Dx'):
                        tmp = l.split(': ')[1].split(',')
                        for c in tmp:
                            classes.add(c.strip())
    return sorted(classes)