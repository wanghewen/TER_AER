import dgl
import dgl.function as fn
from dgl.data import DGLDataset
import torch
import CommonModules as CM
import os
import collections
from sklearn.model_selection import train_test_split
import scipy.io

from datasets.base.utils import get_line_graph
from global_parameters import YelpchiLineGraphDataFolder, YelpchiEdgeToNodeDataFolder
import numpy as np
import pandas as pd
from icecream import ic


# # Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# # And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
# class YelpchiLineGraph(DGLDataset):
#     def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False):
#         self.homo = homo
#         self.aggregate = aggregate
#         super().__init__(name='YelpchiProcessed')
#         if subset == "train":
#             self.nodes_mask = self.graph.ndata["train_mask"]
#         elif subset == "val":
#             self.nodes_mask = self.graph.ndata["val_mask"]
#         elif subset == "test":
#             self.nodes_mask = self.graph.ndata["test_mask"]
#         else:
#             self.nodes_mask = None
#         if no_node_features:  # Exclude any node features
#             self.nfeats = torch.eye(self.graph.num_nodes())
#         else:
#             self.nfeats = self.graph.ndata["nfeat"]
#
#     #         self.efeats = self.graph.edata["efeat"]
#
#     def process(self):
#         data = scipy.io.loadmat(os.path.join(YelpchiLineGraphDataFolder, 'YelpChi.mat'))
#         adj, embeds, labels = data["net_rur"], data["features"], data["label"]
#         #         adj, embeds, labels = data["homo"], data["features"], data["label"]
#         n_nodes = adj.shape[0]
#         embeds = torch.from_numpy(embeds.A.astype(np.float32))
#         labels = torch.from_numpy(labels[0])
#         index = torch.arange(n_nodes)
#         self.graph = dgl.from_scipy(adj)
#         train_index, test_index = train_test_split(index, test_size=0.2, random_state=12)
#         train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=12)
#         train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         test_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         train_mask[train_index] = 1
#         val_mask[val_index] = 1
#         test_mask[test_index] = 1
#
#         self.graph.ndata["nfeat"] = embeds
#         self.graph.ndata["label"] = labels
#         self.graph.ndata['train_mask'] = train_mask
#         self.graph.ndata['val_mask'] = val_mask
#         self.graph.ndata['test_mask'] = test_mask
#         self.graph = self.graph.add_self_loop()
#
#     #         if self.homo:
#     #             self.graph = dgl.to_homogeneous(self.graph, ndata=["nfeat", "label", 'train_mask', 'val_mask', 'test_mask']).add_self_loop()
#     #         else:
#     #             pass
#
#     def __getitem__(self, i):
#         if self.aggregate:
#             return {
#                 "g": self.graph,
#                 #             "efeats": self.efeats,
#                 "nfeats": self.nfeats,
#                 "labels": self.graph.ndata["label"],
#                 "nodes_mask": self.nodes_mask,
#                 "edges_id": self.edges_id,
#                 "old_labels": [self.old_train_labels, self.old_val_labels, self.old_test_labels],
#                 "label_index_mapper": [self.train_label_index_mapper, self.val_label_index_mapper,
#                                        self.test_label_index_mapper]
#             }
#         else:
#             return {
#                 "g": self.graph,
#                 #             "efeats": self.efeats,
#                 "nfeats": self.nfeats,
#                 "labels": self.graph.ndata["label"],
#                 "nodes_mask": self.nodes_mask,
#                 #                 "edges_id": self.edges_id,
#                 #             "edges_src": self.edges_src,
#                 #             "edges_dst": self.edges_dst
#             }
#
#     def __len__(self):
#         return 1


class YelpchiLineGraph(DGLDataset):
    # 38264 nodes, 67395 edges -> 67395 nodes, 67395 edges (as directed graph) or 67395 nodes, 48443369 edges
    # (as undirected graph)
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, bidirected=True, initialized_instance=None):
        self.homo = homo
        self.aggregate = aggregate
        self.bidirected = bidirected
        if initialized_instance is None:
            super().__init__(name="YelpchiLineGraph")
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
        df_train = pd.read_csv(os.path.join(YelpchiEdgeToNodeDataFolder, "train_edges.csv"))
        df_val = pd.read_csv(os.path.join(YelpchiEdgeToNodeDataFolder, "val_edges.csv"))
        df_test = pd.read_csv(os.path.join(YelpchiEdgeToNodeDataFolder, "test_edges.csv"))
        train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(YelpchiEdgeToNodeDataFolder, "train_embeds_labels.pkl"))
        val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(YelpchiEdgeToNodeDataFolder, "val_embeds_labels.pkl"))
        test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(YelpchiEdgeToNodeDataFolder, "test_embeds_labels.pkl"))
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
        ic(edge_labels.sum(0), edge_labels.shape[0], "unbalance ratio:", edge_labels.sum(0)/edge_labels.shape[0])
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
        self.graph = get_line_graph(self.graph, as_directed=False, copy=False)
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
    dataset = YelpchiLineGraph()
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)