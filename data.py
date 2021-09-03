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


def na_dataloader(args):
    
    if args.mode == 'not_perturbed': 
        # for douban dataset
        G1, G2= loadG(args.data_folder, args.graphname)           
      
        if args.graphname == 'am-td': # will be revised later 
        
            source_dataset = Dataset('dataset\\DataProcessing\\allmv_tmdb\\allmv')
            target_dataset = Dataset('dataset\\DataProcessing\\allmv_tmdb\\tmdb')
            gt_dr = 'dataset\\DataProcessing\\allmv_tmdb\\dictionaries\\groundtruth'
            gt_dict = graph_utils.load_gt(gt_dr, source_dataset.id2idx, target_dataset.id2idx, 'dict')
            gt_dict = DeleteDuplicatedElement(gt_dict)
    
            attr1, attr2, attribute_sim  = AttributeProcessing(args, G1, G2, gt_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, gt_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict

            
        else:        
            
            alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)
            attr1, attr2, attribute_sim = AttributeProcessing(args, G1, G2, alignment_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, alignment_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict

            
    else: 
        # for perturbed
        G1, G2 = loadG(args.data_folder, args.graphname)
        alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)
        G3, G4, Ggt = PerturbedProcessing(G1, G2, 0, 0.05, args.graphname)
        
        attr1, attr2, attribute_sim = AttributeProcessing(args, G3, G4, alignment_dict)    
        #alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)  
        idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
        
        return G3, G4, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict
