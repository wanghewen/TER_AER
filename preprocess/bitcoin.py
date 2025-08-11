import os
import pandas as pd
import networkx as nx
import dgl
import scipy.sparse
import torch
import CommonModules as CM
from global_parameters import BitcoinGraphDataFolder, BitcoinHeterogeneousGraphDataFolder, \
    BitcoinHomogeneousGraphDataFolder, BitcoinHeterogeneousWithnoneGraphDataFolder, BitcoinHomogeneousWithnoneGraphDataFolder

days = ['2016-03-11',
        '2016-03-12',
        '2016-03-13',
        '2016-03-14',
        '2016-03-15',
        '2016-03-16',
        '2016-03-17',
        '2016-03-18',
        '2016-03-19',
        '2016-03-20',
        '2016-03-21',
        '2016-03-22',
        '2016-03-23',
        '2016-03-24']


def main(is_homogeneous=False, withnone=False):
    if is_homogeneous:
        os.makedirs(BitcoinHomogeneousGraphDataFolder, exist_ok=True)
    else:
        os.makedirs(BitcoinHeterogeneousGraphDataFolder, exist_ok=True)
    dgl_graphs = []
    lambert_features = CM.IO.ImportFromPkl("/data/Bitcoin/2016_biweek/df_2016_biweek_dict_expanded_label.pickle")
    day_train_address_dict, day_val_address_dict, day_test_address_dict = CM.IO.ImportFromPkl(
        "/data/Bitcoin/2016_biweek/2016_biweek_day_address_dict_split_by_time.pickle")
    # day = days[0]
    for day in days:
        print("Processing day", day)
        G, _, _, _ = CM.IO.ImportFromPkl(os.path.join(BitcoinGraphDataFolder, day + ".pkl"))

        G_address = pd.DataFrame(list(G.nodes()), columns=['address'])
        train_address = pd.DataFrame(day_train_address_dict[day], columns=['address'])
        train_address["train_mask"] = 1
        val_address = pd.DataFrame(day_val_address_dict[day], columns=['address'])
        val_address["val_mask"] = 1
        test_address = pd.DataFrame(day_test_address_dict[day], columns=['address'])
        test_address["test_mask"] = 1
        G_leftjoin = G_address.set_index('address').join(lambert_features[day].set_index('address'), how = 'left').join(train_address.set_index('address'), how = 'left').join(val_address.set_index('address'), how = 'left').join(test_address.set_index('address'), how = 'left')
        # print(G_leftjoin["train_mask"].sum())
        if withnone:  # Only assign 0 on train set for label None (to make data more balance during training)
            G_leftjoin.loc[pd.isna(G_leftjoin["label"]), ["train_mask"]] = 0
        else:
            G_leftjoin.loc[pd.isna(G_leftjoin["label"]), ["train_mask", "val_mask", "test_mask"]] = 0
        features = G_leftjoin.drop(columns=["label", "train_mask", "val_mask", "test_mask"]).fillna(0).values
        node_label = G_leftjoin["label"].fillna("white").values
        node_label[pd.isna(node_label)] = 'white'
        node_label[node_label != 'white'] = 1
        node_label[node_label != 1] = 0

        adj = nx.to_scipy_sparse_matrix(G, nodelist=(G.nodes()))
        if is_homogeneous:
            adj: scipy.sparse.csr_matrix = adj @ adj
            adj.data[:] = 1
        dgl_graph: dgl.DGLGraph = dgl.from_scipy(adj)
        dgl_graph.ndata["nfeat"] = torch.FloatTensor(features)
        print(G_leftjoin["train_mask"].sum())
        dgl_graph.ndata["train_mask"] = torch.LongTensor(G_leftjoin["train_mask"].fillna(0))
        dgl_graph.ndata["val_mask"] = torch.LongTensor(G_leftjoin["val_mask"].fillna(0))
        dgl_graph.ndata["test_mask"] = torch.LongTensor(G_leftjoin["test_mask"].fillna(0))
        dgl_graph.ndata["label"] = torch.LongTensor(node_label.astype(int))
        dgl_graphs.append(dgl_graph)

    if is_homogeneous:
        if withnone:
            dgl.save_graphs(os.path.join(BitcoinHomogeneousWithnoneGraphDataFolder, "graph.bin"), dgl_graphs)
        else:
            dgl.save_graphs(os.path.join(BitcoinHomogeneousGraphDataFolder, "graph.bin"), dgl_graphs)
    else:
        if withnone:
            dgl.save_graphs(os.path.join(BitcoinHeterogeneousWithnoneGraphDataFolder, "graph.bin"), dgl_graphs)
        else:
            dgl.save_graphs(os.path.join(BitcoinHeterogeneousGraphDataFolder, "graph.bin"), dgl_graphs)
    return


if __name__ == "__main__":
    main(is_homogeneous=True, withnone=False)
    # main(is_homogeneous=False, withnone=True)