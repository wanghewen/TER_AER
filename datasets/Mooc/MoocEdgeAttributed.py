from datasets.base.EdgeAttributed import EdgeAttributed


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
from global_parameters import MOOCCountVectorizerDataFolder


class MoocEdgeAttributed(EdgeAttributed):
    #  nodes,  edges ->  nodes,  edges
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, subset=None, no_node_features=False, initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, **kwargs):
        # AmazonEdgeToNodeDataFolder = "/data/PublicGraph/AmazonReview/data_sentence_transformer"
        data_folder = MOOCCountVectorizerDataFolder
        super().__init__(name="MoocEdgeAttributed", data_folder=data_folder,
                         subset=subset, no_node_features=no_node_features,
                         initialized_instance=initialized_instance,
                         add_self_loop=add_self_loop,
                         randomize_train_test=randomize_train_test,
                         randomize_by_node=randomize_by_node, **kwargs)


if __name__ == "__main__":
    dataset = MoocEdgeAttributed(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph, graph["g"].is_multigraph)