import CommonModules as CM
import os
import pandas as pd

from datasets.base.EdgeToNode import EdgeToNode
from global_parameters import TransactionEdgeToNodeDataFolder
from icecream import ic


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class TransactionEdgeToNode(EdgeToNode):
    # 24141 nodes, 141141 edges -> 165282 nodes, 729846 edges
    @classmethod
    def get_augment_enode_count(cls):
        """
        Upper limit for initialize augmented enode features. Must be greater than or equal to the number of positive nodes in the training set.
        :return:
        """
        return 4321*2

    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, end_node_same_type=True, randomize_by_node=False):
        data_folder = TransactionEdgeToNodeDataFolder
        super().__init__(name="TransactionEdgeToNode", data_folder=data_folder, subset=subset,
                         no_node_features=no_node_features, homo=homo,
                         aggregate=aggregate, initialized_instance=initialized_instance,
                         randomize_train_test=randomize_train_test, bidirectional=bidirectional,
                         end_node_same_type=end_node_same_type, randomize_by_node=randomize_by_node)

if __name__ == "__main__":
    dataset = TransactionEdgeToNode(subset="train", bidirectional=True)
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)