import torch_geometric.utils.convert as cv
from torch_geometric.data import NeighborSampler as RawNeighborSampler

import pandas as pd
from utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import collections
import networkx as nx
import copy 
from sklearn.metrics import roc_auc_score
import os
from models import *
import numpy as np
import random
import torch
from data import *

def set_seeds(n):
    seed = int(n)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 12
set_seeds(seed)
print("set seed:", seed)
#%% arguments
def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run myProject.")
    parser.add_argument('--attribute_folder', nargs='?', default='dataset/attribute/')
    parser.add_argument('--data_folder', nargs='?', default='dataset/graph/')
    parser.add_argument('--alignment_folder', nargs='?', default='dataset/alignment/',
                         help="Make sure the alignment numbering start from 0")
    parser.add_argument('--k_hop', nargs='?', default=2)  
    parser.add_argument('--train_ratio', nargs='?', default= 0.1) 
    parser.add_argument('--graphname', nargs='?', default='fb-tt')  #fb-tt attribute 좀 이상한데? 0이 잇네 몇개
    parser.add_argument('--mode', nargs='?', default='not_perturbed', help="not_perturbed or perturbed or real_perturbed") 
    
    return parser.parse_args()

args = parse_args()


''' ------------------------------------------------------------- main -------------------------------------------------------------  '''
#%% main script

G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict = na_dataloader(args)
GradAlign = GradAlign(G1, G2, attr1, attr2, args.k_hop, alignment_dict, alignment_dict_reversed, \
                          args.train_ratio, idx1_dict, idx2_dict)   
S_mv, S_prime, seed_list1, seed_list2 = GradAlign.run_algorithm()



    
    
    



