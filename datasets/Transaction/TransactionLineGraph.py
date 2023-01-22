import dgl
import dgl.function as fn
import scipy.sparse
from dgl.data import DGLDataset
import torch
import CommonModules as CM
import os
import collections
import numpy as np
import pandas as pd

from datasets.base.utils import get_line_graph
from global_parameters import TransactionEdgeToNodeDataFolder
from icecream import ic
import networkx as nx


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class TransactionLineGraph(DGLDataset):
    # 24141 nodes, 141141 edges -> 141141 nodes, 38852075 edges (as directed graph) or 141141 nodes, 159643805 edges
    # (as undirected graph)
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, bidirected=True, initialized_instance=None):
        self.homo = homo
        self.aggregate = aggregate
        self.bidirected = bidirected
        if initialized_instance is None:
            super().__init__(name="TransactionLineGraph")
        else:
            self = initialized_instance
        if subset == "train":
            self.nodes_mask = self.graph.ndata["train_mask"]
        elif subset == "val":
            self.nodes_mask = self.graph.ndata["val_mask"]
        elif subset == "test":
            self.nodes_mask = self.graph.ndata["test_mask"]
        else:
            self.nodes_mask = None
        if no_node_features:  # Exclude any node features
            self.nfeats = torch.eye(self.graph.num_nodes())
        else:
            self.nfeats = self.graph.ndata["nfeat"]
            ic(self.nfeats.shape)
    #         self.efeats = self.graph.edata["efeat"]

    def process(self):
        df_train = pd.read_csv(os.path.join(TransactionEdgeToNodeDataFolder, "train_edges.csv"))
        df_val = pd.read_csv(os.path.join(TransactionEdgeToNodeDataFolder, "val_edges.csv"))
        df_test = pd.read_csv(os.path.join(TransactionEdgeToNodeDataFolder, "test_edges.csv"))
        train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(TransactionEdgeToNodeDataFolder, "train_embeds_labels.pkl"))
        val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(TransactionEdgeToNodeDataFolder, "val_embeds_labels.pkl"))
        test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(TransactionEdgeToNodeDataFolder, "test_embeds_labels.pkl"))
        train_embeds = train_embeds.astype(np.float32)
        val_embeds = val_embeds.astype(np.float32)
        test_embeds = test_embeds.astype(np.float32)
        train_labels[train_labels > 0] = 1
        val_labels[val_labels > 0] = 1
        test_labels[test_labels > 0] = 1
        #         print(type(train_embeds))

        self.df = df = pd.concat([df_train, df_val, df_test], axis=0)
        train_embeds = torch.from_numpy(train_embeds)
        val_embeds = torch.from_numpy(val_embeds)
        test_embeds = torch.from_numpy(test_embeds)
        train_labels = torch.from_numpy(train_labels)
        val_labels = torch.from_numpy(val_labels)
        test_labels = torch.from_numpy(test_labels)
        train_labels = train_labels.reshape(-1, 1)
        val_labels = val_labels.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)
        edge_features = torch.cat([train_embeds, val_embeds, test_embeds])
        edge_labels = torch.cat([train_labels, val_labels, test_labels])
        edge_labels = edge_labels.type(torch.long)
        ic(edge_labels.sum(0), edge_labels.shape[0], "unbalance ratio:", edge_labels.sum(0) / edge_labels.shape[0])
        #         nodes_data = pd.read_csv('./members.csv')
        #         node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        #         node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        #         edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy())
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst))
        # graph = nx.convert_matrix.from_pandas_edgelist(df, source="SENDER_id", target="RECEIVER_id", create_using=nx.MultiGraph)

        n_train = train_embeds.shape[0]
        n_val = val_embeds.shape[0]
        n_test = test_embeds.shape[0]
        n_nodes = self.graph.number_of_edges()
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        edge_start_index = 0
        train_mask[edge_start_index:edge_start_index + n_train] = True
        val_mask[edge_start_index + n_train:edge_start_index + n_train + n_val] = True
        test_mask[edge_start_index + n_train + n_val:edge_start_index + n_train + n_val + n_test] = True
        # After converting into line graph, edge features will be node features
        self.graph.edata["nfeat"] = edge_features
        self.graph.edata["train_mask"] = train_mask
        self.graph.edata["val_mask"] = val_mask
        self.graph.edata["test_mask"] = test_mask
        self.graph.edata["label"] = edge_labels
        # G = dgl.graph(([0,1,2],[1,2,1]))
        # get_line_graph(G, False)
        self.graph = get_line_graph(self.graph, as_directed=True, copy=False)
        self.graph = self.graph.add_self_loop()

    def __getitem__(self, i):
        return {
            "g": self.graph,
            #             "efeats": self.efeats,
            "nfeats": self.nfeats,
            "labels": self.graph.ndata["label"],
            "nodes_mask": self.nodes_mask,
            # "edges_id": self.edges_id,
            #             "edges_src": self.edges_src,
            #             "edges_dst": self.edges_dst
        }

    def __len__(self):
        return 1


if __name__ == "__main__":
    dataset = TransactionLineGraph(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)