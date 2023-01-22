import CommonModules as CM
import os

import dgl
import numpy as np
import pandas as pd
import scipy.sparse
import torch

from datasets.base.EdgeAttributed import EdgeAttributed


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class Citation(EdgeAttributed):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, name, subset=None, no_node_features=False, initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, num_classes=None, **kwargs):
        self.num_classes = num_classes  # Limit the number of classes for multilabel classification
        super().__init__(name=name, subset=subset, no_node_features=no_node_features,
                         initialized_instance=initialized_instance, add_self_loop=add_self_loop,
                         randomize_train_test=randomize_train_test,
                         randomize_by_node=randomize_by_node, num_classes=num_classes, **kwargs)
