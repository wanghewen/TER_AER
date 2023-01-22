import os

import dgl

from datasets.base.EdgeAttributed import EdgeAttributed
from global_parameters import DBLPCountVectorizerDataFolder


class DBLPEdgeAttributed(EdgeAttributed):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, num_classes=10, subset=None, no_node_features=False,
                 initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, sample_size=None,
                 sample_way="bfs", reduced_dim=None, **kwargs):
        data_folder = DBLPCountVectorizerDataFolder
        # data_folder = "/data/PublicGraph/DBLP/data_count_vectorizer"
        # if initialized_instance is None:
        #     self.df_train = pd.read_csv(os.path.join(data_folder, "train_edges.csv"))
        #     self.df_val = pd.read_csv(os.path.join(data_folder, "val_edges.csv"))
        #     self.df_test = pd.read_csv(os.path.join(data_folder, "test_edges.csv"))
        #     self.train_embeds, self.train_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "train_embeds_labels.pkl"))
        #     self.val_embeds, self.val_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "val_embeds_labels.pkl"))
        #     self.test_embeds, self.test_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "test_embeds_labels.pkl"))
        super().__init__(name="DBLPEdgeAttributed", data_folder=data_folder,
                         subset=subset, no_node_features=no_node_features,
                         initialized_instance=initialized_instance, add_self_loop=add_self_loop,
                         randomize_train_test=randomize_train_test,
                         randomize_by_node=randomize_by_node, num_classes=num_classes,
                         sample_way=sample_way, sample_size=sample_size, reduced_dim=reduced_dim, **kwargs)

if __name__ == "__main__":
    dataset = DBLPEdgeAttributed(subset="train", add_self_loop=False, randomize_train_test=False,
                             randomize_by_node=False, sample_size=5000,
                             reduced_dim=256)
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)
    print(graph["labels"][graph["edges_mask"]].sum())
    u, v = graph["g"].edges()
    # print(set(u.tolist()) - set(v.tolist()))
    u += 1
    v += 1
    with open("/home/wanghw/pycharm/DataAugmentation/models/Edge2vec/DBLP-5K.graph", "w") as fd:
        for u, v in zip(u.tolist(), v.tolist()):
            print(u, v, file=fd)
