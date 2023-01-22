import dgl
import dgl.function as fn
from dgl.data import DGLDataset
import torch
import CommonModules as CM
import os
import collections
import numpy as np
import pandas as pd

from datasets.base.EdgeAttributed import EdgeAttributed
from global_parameters import YelpchiEdgeToNodeDataFolder
from icecream import ic

# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class YelpchiEdgeAttributed(EdgeAttributed):
    # 38264 nodes, 67395 edges -> 38264 nodes, 105659 edges
    @classmethod
    def get_node_count(cls):
        return 38264

    def __init__(self, subset=None, no_node_features=False, initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, **kwargs):
        # This will introduce zero attributed edges, which will affect training process like batch normalization
        self.add_self_loop = add_self_loop
        self.randomize_train_test = randomize_train_test
        self.randomize_by_node = randomize_by_node
        data_folder = YelpchiEdgeToNodeDataFolder
        if initialized_instance is None:
            super().__init__(name="YelpchiEdgeAttributed", subset=subset,
                             data_folder=data_folder,
                             no_node_features=no_node_features,
                             initialized_instance=initialized_instance, add_self_loop=add_self_loop,
                             randomize_train_test=randomize_train_test, randomize_by_node=randomize_by_node, **kwargs)
        else:
            self = initialized_instance
        if subset == "train":
            self.edges_mask = self.graph.edata["train_mask"]
        elif subset == "val":
            self.edges_mask = self.graph.edata["val_mask"]
        elif subset == "test":
            self.edges_mask = self.graph.edata["test_mask"]
        else:
            self.edges_mask = None
        if no_node_features:  # Exclude any node features
            self.nfeats = None
        else:
            self.nfeats = self.graph.ndata["nfeat"]

        self.efeats = self.graph.edata["efeat"]


if __name__ == "__main__":
    dataset = YelpchiEdgeAttributed(subset="train", randomize_by_node=True)
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print((graph["g"].edata["efeat"] != 0).sum())
    print(graph)