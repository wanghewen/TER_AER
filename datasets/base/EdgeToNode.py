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
from sklearn.model_selection import train_test_split

from global_parameters import YelpchiEdgeToNodeDataFolder
from icecream import ic

# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class EdgeToNode(DGLDataset):
    def __init__(self, name, data_folder, subset=None, no_node_features=False, homo=True, aggregate=False,
                 initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, end_node_same_type=True, randomize_by_node=False,
                 num_classes=None, sample_way=None, sample_size=None, reduced_dim=None):
        self.homo = homo  # Not useful right now
        self.aggregate = aggregate
        self.randomize_train_test = randomize_train_test
        self.randomize_by_node = randomize_by_node
        self.bidirectional = bidirectional  # Whether to use directed edge or not for edge to node graph
        self.end_node_same_type = end_node_same_type  # Whether to treat end node as same type. If not, it will re-encode end nodes as different nodes
        self.num_classes = num_classes
        if initialized_instance is None:
            if reduced_dim:
                data_folder = os.path.join(data_folder, f"reduce_dim_{reduced_dim}")
            if sample_size:
                data_folder = data_folder + f"_{sample_way}_sampled_size={sample_size}"
            else:
                data_folder = data_folder
            self.df_train = pd.read_csv(os.path.join(data_folder, "train_edges.csv"))
            self.df_val = pd.read_csv(os.path.join(data_folder, "val_edges.csv"))
            self.df_test = pd.read_csv(os.path.join(data_folder, "test_edges.csv"))
            self.train_embeds, self.train_labels = CM.IO.ImportFromPkl(
                os.path.join(data_folder, "train_embeds_labels.pkl"))
            self.val_embeds, self.val_labels = CM.IO.ImportFromPkl(
                os.path.join(data_folder, "val_embeds_labels.pkl"))
            self.test_embeds, self.test_labels = CM.IO.ImportFromPkl(
                os.path.join(data_folder, "test_embeds_labels.pkl"))
            if len(self.train_labels.shape) > 1:
                label_column_mask = np.array(self.train_labels.sum(0) > 10).reshape(-1)  # Exclude class with too few
                # samples
                self.train_labels = self.train_labels[:, label_column_mask]  # Exclude class with too few samples
                self.val_labels = self.val_labels[:, label_column_mask]  # Exclude class with too few samples
                self.test_labels = self.test_labels[:, label_column_mask]  # Exclude class with too few samples
            else:
                self.train_labels = self.train_labels.reshape(-1, 1)
                self.val_labels = self.val_labels.reshape(-1, 1)
                self.test_labels = self.test_labels.reshape(-1, 1)
            super().__init__(name=name)
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

    #         self.efeats = self.graph.edata["efeat"]

    def _process_aggregate(self, df, embeds, labels):
        sender_trans_dict = collections.defaultdict(list)  # Record original df item details
        sender_trans_index_dict = collections.defaultdict(list)  # Record original df row index
        sender_trans_emb_dict = collections.defaultdict(list)
        sender_trans_label_dict = collections.defaultdict(list)
        sender_trans_count_dict = collections.defaultdict(list)
        sender_id_list = []
        receiver_id_list = []
        item_details_list = []
        sentence_index_list = []
        sentence_label_list = []
        label_index_mapper = []  # Used to convert back to ground truth labels
        for row_index, row in df.iterrows():
            sender_id = row['SENDER_id']
            receiver_id = row['RECEIVER_id']
            item_details = row['item_details']
            sentence_index = -1
            for sentence_index, sentence in enumerate(sender_trans_dict[sender_id]):
                if fuzz.partial_ratio(item_details, sentence) > 80:
                    sender_trans_emb_dict[sender_id][sentence_index] = (sender_trans_emb_dict[sender_id][
                                                                            sentence_index] *
                                                                        sender_trans_count_dict[sender_id][
                                                                            sentence_index] + embeds[row_index]) / (
                                                                                   sender_trans_count_dict[sender_id][
                                                                                       sentence_index] + 1)
                    #                     sender_trans_label_dict[sender_id][sentence_index] = sender_trans_label_dict[sender_id][sentence_index] # Can update label if you want
                    sender_trans_count_dict[sender_id][sentence_index] += 1
                    label_index_mapper.append(sender_trans_index_dict[sender_id][sentence_index])
                    break
            else:
                sender_trans_dict[sender_id].append(item_details)
                sender_trans_index_dict[sender_id].append(len(sentence_label_list))
                sender_trans_label_dict[sender_id].append(labels[row_index])
                sender_trans_emb_dict[sender_id].append(embeds[row_index])
                sender_trans_count_dict[sender_id].append(1)
                sender_id_list.append(sender_id)
                receiver_id_list.append(receiver_id)
                item_details_list.append(item_details)
                sentence_index_list.append(sentence_index + 1)
                sentence_label_list.append(labels[row_index])
                label_index_mapper.append(len(sentence_label_list) - 1)
        #                 item_details_index_list.append(row_index)
        # Generate new df/edge features/labels
        print("Removed", df.shape[0] - len(sender_id_list), "transactions through aggregation")
        df = pd.DataFrame(
            {'SENDER_id': sender_id_list, 'RECEIVER_id': receiver_id_list, 'item_details': item_details_list})
        new_embeds = []
        new_labels = []
        for row_index, row in df.iterrows():
            sender_id = row['SENDER_id']
            new_embeds.append(sender_trans_emb_dict[sender_id][sentence_index_list[row_index]])
            new_labels.append(sentence_label_list[row_index])
        embeds = np.array(new_embeds, dtype=embeds.dtype)
        labels = np.array(new_labels, dtype=labels.dtype)
        return df, embeds, labels, label_index_mapper

    def process(self):
        print("Number of edges for original graph:", self.df_train.shape[0] + self.df_val.shape[0] + self.df_test.shape[0])
        print("Number of Nodes for original graph:",
              len(set(self.df_train['SENDER_id']) | set(self.df_train['RECEIVER_id']) |
                 set(self.df_val['SENDER_id']) | set(self.df_val['RECEIVER_id']) |
                  set(self.df_test['SENDER_id']) | set(self.df_test['RECEIVER_id'])))
        if self.randomize_train_test and not self.randomize_by_node:
            print("Randomly split dataset by edges...")
            df = pd.concat([self.df_train, self.df_val, self.df_test])
            if isinstance(self.train_embeds, (np.ndarray, torch.Tensor)):
                embeds = np.concatenate([self.train_embeds, self.val_embeds, self.test_embeds], axis=0)
            else:
                embeds = scipy.sparse.vstack([self.train_embeds, self.val_embeds, self.test_embeds])
            if isinstance(self.train_labels, (np.ndarray, torch.Tensor)):
                labels = np.concatenate([self.train_labels, self.val_labels, self.test_labels], axis=0)
            else:
                labels = scipy.sparse.vstack([self.train_labels, self.val_labels, self.test_labels])
            if self.num_classes:
                labels = self.labels[:, :self.num_classes].A
                self.labels = labels
            index = np.arange(df.shape[0])
            train_index, test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
            train_index, val_index = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=12)
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            train_embeds, val_embeds, test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            train_labels, val_labels, test_labels = labels[train_index], labels[val_index], labels[test_index]
        elif self.randomize_by_node:
            print("Randomly split dataset by nodes...")
            df = pd.concat([self.df_train, self.df_val, self.df_test])
            if isinstance(self.train_embeds, (np.ndarray, torch.Tensor)):
                embeds = np.concatenate([self.train_embeds, self.val_embeds, self.test_embeds], axis=0)
            else:
                embeds = scipy.sparse.vstack([self.train_embeds, self.val_embeds, self.test_embeds])
            if isinstance(self.train_labels, (np.ndarray, torch.Tensor)):
                labels = np.concatenate([self.train_labels, self.val_labels, self.test_labels], axis=0)
            else:
                labels = scipy.sparse.vstack([self.train_labels, self.val_labels, self.test_labels])
            if self.num_classes:
                labels = self.labels[:, :self.num_classes].A
                self.labels = labels
            ###########DEBUG############
            # df["labels"] = labels
            # df = df.drop_duplicates(subset=['SENDER_id', 'labels'], ignore_index=False)
            # embeds = embeds[df.index]
            # labels = labels[df.index]
            ############################
            all_node_list = np.array(list(set(df["SENDER_id"]) | set(df["RECEIVER_id"])))
            index = np.arange(len(all_node_list))
            node_train_index, node_test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
            node_train_index, node_val_index = train_test_split(node_train_index, test_size=0.2, shuffle=True, random_state=12)
            train_index = (df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_train_index])).to_numpy()
            val_index = ((df["SENDER_id"].isin(all_node_list[node_val_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_val_index])) & ~train_index).to_numpy()
            test_index = ((df["SENDER_id"].isin(all_node_list[node_test_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_test_index])) & ~train_index).to_numpy()
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            train_embeds, val_embeds, test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            train_labels, val_labels, test_labels = labels[train_index], labels[val_index], labels[test_index]
        else:
            print("Not randomly split dataset...")
            df_train, df_val, df_test = self.df_train, self.df_val, self.df_test
            train_embeds, val_embeds, test_embeds = self.train_embeds, self.val_embeds, self.test_embeds
            if self.num_classes:
                self.train_labels = self.train_labels[:, :self.num_classes].A
                self.val_labels = self.val_labels[:, :self.num_classes].A
                self.test_labels = self.test_labels[:, :self.num_classes].A
            train_labels, val_labels, test_labels = self.train_labels, self.val_labels, self.test_labels
        # TODO: Check if embeds are sparse

        train_embeds = train_embeds.astype(np.float32)
        val_embeds = val_embeds.astype(np.float32)
        test_embeds = test_embeds.astype(np.float32)
        if scipy.sparse.isspmatrix(train_embeds):
            train_embeds = train_embeds.A
            val_embeds = val_embeds.A
            test_embeds = test_embeds.A
        if scipy.sparse.isspmatrix(train_labels):
            train_labels = train_labels.A
            val_labels = val_labels.A
            test_labels = test_labels.A
        train_labels[train_labels > 0] = 1
        val_labels[val_labels > 0] = 1
        test_labels[test_labels > 0] = 1

        train_labels[train_labels < 0] = 0
        val_labels[val_labels < 0] = 0
        test_labels[test_labels < 0] = 0
        #         print(type(train_embeds))
        if self.aggregate:
            old_train_labels = train_labels
            old_val_labels = val_labels
            old_test_labels = test_labels
            df_train, train_embeds, train_labels, train_label_index_mapper = self._process_aggregate(df_train,
                                                                                                          train_embeds,
                                                                                                          train_labels)
            df_val, val_embeds, val_labels, val_label_index_mapper = self._process_aggregate(df_val, val_embeds,
                                                                                                  val_labels)
            df_test, test_embeds, test_labels, test_label_index_mapper = self._process_aggregate(df_test,
                                                                                                      test_embeds,
                                                                                                      test_labels)

        self.df = df = pd.concat([df_train, df_val, df_test], axis=0)
        train_embeds = torch.from_numpy(train_embeds)
        val_embeds = torch.from_numpy(val_embeds)
        test_embeds = torch.from_numpy(test_embeds)
        train_labels = torch.from_numpy(train_labels)
        val_labels = torch.from_numpy(val_labels)
        test_labels = torch.from_numpy(test_labels)
        edge_features = torch.cat([train_embeds, val_embeds, test_embeds])
        edge_labels = torch.cat([train_labels, val_labels, test_labels]).long()

        #         nodes_data = pd.read_csv('./members.csv')
        #         node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        #         node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        #         edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy())
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy())
        if self.end_node_same_type:
            sender_key = receiver_key = "user"
        else:
            edges_src = edges_src.unique(return_inverse=True)[1]  # Re-encode SENDER_id to start from 1
            edges_dst = edges_dst.unique(return_inverse=True)[1]  # Re-encode RECEIVER_id to start from 1
            sender_key = "sender"
            receiver_key = "receiver"
        # self.edge_start_index = max(max(edges_src), max(edges_dst)) + 1
        edge_node_index = torch.arange(edge_features.shape[0])
        #         if self.homo:
        self.graph: dgl.DGLGraph
        if self.bidirectional:
            self.graph = dgl.heterograph({(sender_key, "send", "transaction"): (edges_src, edge_node_index),
                                          ("transaction", "sent-by", sender_key): (edge_node_index, edges_src),
                                          (receiver_key, "receive", "transaction"): (edges_dst, edge_node_index),
                                          ("transaction", "received-by", receiver_key): (edge_node_index, edges_dst)
                                          })
        else:
            self.graph = dgl.heterograph({(sender_key, "send", "transaction"): (edges_src, edge_node_index),
                                          ("transaction", "received-by", receiver_key): (edge_node_index, edges_dst)
                                          })
        #         self.graph = dgl.graph((edge_node_index, edges_dst))
        assert (self.graph.ntypes == ["receiver", "sender", "transaction"]) or (self.graph.ntypes == ["transaction", "user"])
        print(edge_features.shape, edges_src.shape, edge_node_index.shape)
        if self.end_node_same_type:
            self.edge_start_index = max(max(edges_src), max(edges_dst)) + 1
            self.graph.nodes[sender_key].data['nfeat'] = torch.zeros(self.edge_start_index, train_embeds.shape[1])
            self.graph.nodes["transaction"].data['nfeat'] = edge_features
            edge_label_shape = list(edge_labels.shape)
            edge_label_shape[0] = self.edge_start_index
            self.graph.nodes[sender_key].data['label'] = torch.zeros(edge_label_shape, dtype=torch.long)
            self.graph.nodes["transaction"].data['label'] = edge_labels
        else:
            self.graph.nodes[sender_key].data['nfeat'] = torch.zeros(edges_src.max()+1, train_embeds.shape[1])
            self.graph.nodes[receiver_key].data['nfeat'] = torch.zeros(edges_dst.max()+1, train_embeds.shape[1])
            self.graph.nodes["transaction"].data['nfeat'] = edge_features
            edge_label_shape = list(edge_labels.shape)
            edge_label_shape[0] = edges_src.max()+1
            self.graph.nodes[sender_key].data['label'] = torch.zeros(edge_label_shape, dtype=torch.long)
            edge_label_shape = list(edge_labels.shape)
            edge_label_shape[0] = edges_dst.max()+1
            self.graph.nodes[receiver_key].data['label'] = torch.zeros(edge_label_shape, dtype=torch.long)
            self.graph.nodes["transaction"].data['label'] = edge_labels
        #         self.graph.ndata[dgl.NTYPE] = node_types
        #         self.graph.edata['efeat'] = edge_features
        #         self.graph.edata['label'] = edge_labels
        #         self.graph.update_all(fn.copy_e('efeat', 'm'), fn.mean('m', 'nfeat')) # i.e. self.graph.ndata["nfeat"]
        #         self.graph = dgl.add_self_loop(self.graph)  # only 1 graph in dataset
        self.edges_id = torch.arange(self.graph.number_of_edges())
        #         self.edges_src, self.edges_dst = self.graph.find_edges(self.edges_id)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        #         m_edges = self.graph.number_of_edges()
        n_nodes_transaction = self.graph.number_of_nodes(ntype="transaction")
        n_train = train_embeds.shape[0]
        n_val = val_embeds.shape[0]
        n_test = test_embeds.shape[0]
        train_mask = torch.zeros(n_nodes_transaction, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes_transaction, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes_transaction, dtype=torch.bool)
        edge_start_index = 0
        train_mask[edge_start_index:edge_start_index + n_train] = True
        val_mask[edge_start_index + n_train:edge_start_index + n_train + n_val] = True
        test_mask[edge_start_index + n_train + n_val:edge_start_index + n_train + n_val + n_test] = True
        self.graph.nodes["transaction"].data['train_mask'] = train_mask
        self.graph.nodes["transaction"].data['val_mask'] = val_mask
        self.graph.nodes["transaction"].data['test_mask'] = test_mask
        eid = torch.arange(n_nodes_transaction, dtype=torch.int)
        self.graph.nodes["transaction"].data['eid'] = eid
        for ntype in [sender_key, receiver_key]:
            n_nodes_ntype = self.graph.number_of_nodes(ntype=ntype)
            self.graph.nodes[ntype].data['train_mask'] = torch.zeros(n_nodes_ntype, dtype=torch.bool)
            self.graph.nodes[ntype].data['val_mask'] = torch.zeros(n_nodes_ntype, dtype=torch.bool)
            self.graph.nodes[ntype].data['test_mask'] = torch.zeros(n_nodes_ntype, dtype=torch.bool)
            eid = torch.zeros(n_nodes_ntype, dtype=torch.int)
            eid[:] = -1
            self.graph.nodes[ntype].data['eid'] = eid
        if self.homo:
            self.graph = dgl.to_homogeneous(self.graph, ndata=["nfeat", "label", 'train_mask', 'val_mask',
                                                               'test_mask', "eid"])
            # if self.bidirectional:
            self.graph = self.graph.add_self_loop()
            l = self.graph.ndata["eid"][:n_nodes_transaction]
            # Make sure in edge to node graph, the eids for transactions are sorted
            assert all(l[i] <= l[i + 1] for i in range(len(l) - 1))
        else:
            pass

    def __getitem__(self, i):
        if self.aggregate:
            return {
                "g": self.graph,
                #             "efeats": self.efeats,
                "nfeats": self.nfeats,
                "labels": self.graph.ndata["label"],
                "nodes_mask": self.nodes_mask,
                "edges_id": self.edges_id,
                "old_labels": [self.old_train_labels, self.old_val_labels, self.old_test_labels],
                "label_index_mapper": [self.train_label_index_mapper, self.val_label_index_mapper,
                                       self.test_label_index_mapper]
            }
        else:
            return {
                "g": self.graph,
                #             "efeats": self.efeats,
                "nfeats": self.nfeats,
                "labels": self.graph.ndata["label"],
                "nodes_mask": self.nodes_mask,
                "edges_id": self.edges_id,
                #             "edges_src": self.edges_src,
                #             "edges_dst": self.edges_dst
                "end_node_same_type": self.end_node_same_type,
            }

    def __len__(self):
        return 1


# if __name__ == "__main__":
#     dataset = YelpchiEdgeToNode(subset="train")
#     graph = dataset[0]
#     # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
#     print(graph)