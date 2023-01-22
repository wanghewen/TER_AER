import CommonModules as CM
import os
import pandas as pd

from datasets.base.EdgeToNode import EdgeToNode
from global_parameters import MOOCCountVectorizerDataFolder


class MoocEdgeToNode(EdgeToNode):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, randomize_by_node=False, reduced_dim=None):
        data_folder = MOOCCountVectorizerDataFolder
        super().__init__(name="MoocEdgeToNode", data_folder=data_folder, subset=subset,
                         no_node_features=no_node_features, homo=homo, aggregate=aggregate,
                         initialized_instance=initialized_instance, randomize_train_test=randomize_train_test,
                         bidirectional=bidirectional, randomize_by_node=False, reduced_dim=reduced_dim)




if __name__ == "__main__":
    dataset = MoocEdgeToNode(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)