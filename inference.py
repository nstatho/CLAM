from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

parser = argparse.ArgumentParser(description='CLAM Inference Script')
parser.add_argument('--slide-path', type=str, default=None,
                    help='path to slide')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--model_weights', type=str, default=None,
                    help='path to trained model weights')
parser.add_argument('--n_classes', type=int, default=None,
                    help='number of classes for model output')


args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



settings = {'models_dir': args.model_weights,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}


def create_patches():
    pass

def extract_features():
    pass

def predict():
    pass





# initiate database for storing results


# First create the patches

# Second extract the features

# Third predict the label


# model = initiate_model(args, args.model_weights)




# dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
#                         data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
#                         shuffle = False, 
#                         print_info = True,
#                         label_dict = {'normal_tissue':0, 'tumor_tissue':1},
#                         patient_strat=False,
#                         ignore=[])