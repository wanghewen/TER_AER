import os

from datasets.base.EdgeToNode import EdgeToNode
from global_parameters import DBLPCountVectorizerDataFolder


class DBLPEdgeToNode(EdgeToNode):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, num_classes=10, subset=None, no_node_features=False, homo=True, aggregate=False, initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, sample_size=None, sample_way="bfs",
                 randomize_by_node=False, reduced_dim=None):
        data_folder = DBLPCountVectorizerDataFolder
        super().__init__(name="DBLPEdgeToNode", data_folder=data_folder, subset=subset,
                         no_node_features=no_node_features, homo=homo, aggregate=aggregate,
                         initialized_instance=initialized_instance, randomize_train_test=randomize_train_test,
                         bidirectional=bidirectional, randomize_by_node=False, num_classes=num_classes,
                         sample_way=sample_way, sample_size=sample_size, reduced_dim=reduced_dim)




if __name__ == "__main__":
    dataset = DBLPEdgeToNode(subset="train", randomize_train_test=False,
                             randomize_by_node=False, sample_size=10000,
                             reduced_dim=256)
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)
    print(graph["labels"][graph["nodes_mask"]].sum())