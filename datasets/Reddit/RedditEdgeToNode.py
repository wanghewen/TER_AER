from datasets.base.EdgeToNode import EdgeToNode
from global_parameters import RedditCountVectorizerDataFolder


class RedditEdgeToNode(EdgeToNode):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, randomize_by_node=False):
        data_folder = RedditCountVectorizerDataFolder
        super().__init__(name="RedditEdgeToNode", data_folder=data_folder, subset=subset,
                         no_node_features=no_node_features, homo=homo, aggregate=aggregate,
                         initialized_instance=initialized_instance, randomize_train_test=randomize_train_test, bidirectional=bidirectional, randomize_by_node=False)




if __name__ == "__main__":
    dataset = RedditEdgeToNode(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)