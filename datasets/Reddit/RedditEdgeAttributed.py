import CommonModules as CM
import os

import dgl
import numpy as np
import pandas as pd
import scipy.sparse
import torch

from datasets.base.EdgeAttributed import EdgeAttributed
from global_parameters import RedditCountVectorizerDataFolder


class RedditEdgeAttributed(EdgeAttributed):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, subset=None, no_node_features=False, initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, **kwargs):
        # AmazonEdgeToNodeDataFolder = "/data/PublicGraph/AmazonReview/data_sentence_transformer"
        data_folder = RedditCountVectorizerDataFolder
        super().__init__(name="RedditEdgeAttributed", data_folder=data_folder,
                         subset=subset, no_node_features=no_node_features,
                         initialized_instance=initialized_instance,
                         add_self_loop=add_self_loop,
                         randomize_train_test=randomize_train_test,
                         randomize_by_node=randomize_by_node, **kwargs)


if __name__ == "__main__":
    dataset = RedditEdgeAttributed(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph, graph["g"].is_multigraph)