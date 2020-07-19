import numpy as np, os, sys, csv
from scipy.io import loadmat
from models.run import train
from get_12ECG_features import get_12ECG_features

def train_12ECG_classifier(input_directory, output_directory):
    # Get classes and train model
    classes = get_classes(input_directory, train_27=True)
    train(input_directory, output_directory, classes)

    config_vals = ["True", "True", 27]
    with open('checks.txt', 'w') as f:
        for item in config_vals:
            f.write("{}\n".format(str(item)))

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

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