import os

import numpy as np
import scipy.sparse
import torch

from train_EdgeAttributed import args

os.environ["MKL_INTERFACE_LAYER"] = "ILP64"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import pytorch_lightning as pl
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
pl.seed_everything(12)


data_folder = args.data_folder
YelpchiLineGraphDataFolder = os.path.join(data_folder, "Yelpchi/rur_data")
YelpchiEdgeToNodeDataFolder = os.path.join(data_folder, "Yelpchi/edge_to_node_data")
TransactionEdgeToNodeDataFolder = os.path.join(data_folder, 'UDS_data/24k_1239k_onecomponent_remove_duplicate')
FircoEdgeToNodeDataFolder = os.path.join(data_folder, 'Firco')
AMinerCountVectorizerDataFolder = os.path.join(data_folder, "AMiner/data_count_vectorizer_split_by_node")
DBLPCountVectorizerDataFolder = os.path.join(data_folder, "DBLP/data_count_vectorizer_split_by_node")
MOOCCountVectorizerDataFolder = os.path.join(data_folder, "MOOC/data_count_vectorizer_split_by_node")
RedditCountVectorizerDataFolder = os.path.join(data_folder, "Reddit/data_count_vectorizer_split_by_node")
OAGCountVectorizerDataFolder = os.path.join(data_folder, "OAG2.1/data_count_vectorizer_split_by_node")
MINDCountVectorizerDataFolder = os.path.join(data_folder, "MIND/data_count_vectorizer_split_by_node")
