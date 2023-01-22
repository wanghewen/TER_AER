from functools import partial

import dgl
import scipy.sparse
import torch
from dgl import NID
from torch.distributions import Bernoulli
from copy import deepcopy

from datasets.base.utils import get_line_graph


def collate_fn(batch, reset_g_epoch=None):
    # Make sure each epoch has the same graph
    # If the graph is modified during each epoch, remove the line below
    if reset_g_epoch:
        batch[0]["g"] = deepcopy(batch[0]["g"])
    # return batch[0]
    return batch


class NodeAugNeighborSampler(dgl.dataloading.Sampler):
    def __init__(self, fanouts : list[int] = [-1, -1], augmentation_method=None):
        super().__init__()
        self.fanouts = fanouts
        self.augmentation_method = augmentation_method
        self.dist = Bernoulli(0.5)
        self.transforms = dgl.transforms.DropEdge(p=0.5)

    def sample(self, g, seed_nodes):
        # output_nodes = seed_nodes
        subgs_list = []
        subgs_list_aug = []
        # for seed_node in seed_nodes:

        subgs = []
        subgs_aug = []
        # seed_node = [seed_node]
        seed_node = seed_nodes
        g.ndata["seed_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["seed_mask"][seed_node] = True
        if self.augmentation_method:
            exclude_eids = self.dist.sample(torch.Size([g.num_edges()]))
        for fanout in reversed(self.fanouts):
            # Sample a fixed number of neighbors of the current seed nodes.
            sg = g.sample_neighbors(seed_node, fanout)
            # Convert this subgraph to a message flow graph.
            bsg = dgl.to_block(sg, seed_node)
            seed_node = bsg.srcdata[NID]
            subgs.insert(0, sg)
            if self.augmentation_method:
                sg_aug = self.transforms(sg)
                # sg_aug = g.sample_neighbors(seed_node, fanout, exclude_edges=exclude_eids)
                subgs_aug.insert(0, sg_aug)
        # input_nodes = seed_nodes
        # return input_nodes, output_nodes, subgs
        # Get sub graph for batch training
        subg = dgl.merge([dgl.to_block(dgl.merge(subgs), seed_node.tolist())]).to_simple(copy_ndata=True, copy_edata=True)
        # Add self loop for endpoints in the sampled graph
        subg = subg.remove_self_loop()
        subg = subg.add_self_loop()
        # subg = dgl.merge(subgs).to_simple(copy_ndata=True, copy_edata=True)
        subgs_list.append(subg)
        new_g = dgl.batch(subgs_list)  # Or return subgs_list and modify it when training
        batch = {
            "g": new_g,
            "nodes_mask": new_g.ndata["seed_mask"],
            "nfeats": new_g.ndata["nfeat"],
            "labels": new_g.ndata["label"]
        }

        if self.augmentation_method in ["dropedge_subsample"]:
            subg_aug = dgl.merge([dgl.to_block(dgl.merge(subgs_aug), seed_node.tolist())]).to_simple(copy_ndata=True,
                                                                                             copy_edata=True)
            # Add self loop for endpoints in the sampled graph
            subg_aug = subg_aug.remove_self_loop()
            subg_aug = subg_aug.add_self_loop()
            # subg = dgl.merge(subgs).to_simple(copy_ndata=True, copy_edata=True)
            subgs_list_aug.append(subg_aug)
            new_g_aug = dgl.batch(subgs_list_aug)  # Or return subgs_list and modify it when training
            batch_aug = {
                "g": new_g_aug,
                "nodes_mask": new_g_aug.ndata["seed_mask"],
                "nfeats": new_g_aug.ndata["nfeat"],
                "labels": new_g_aug.ndata["label"]
            }

            return batch, batch_aug
        else:
            return batch


class GNNDataloader():
    def __init__(self, reset_g_epoch=False, num_workers=20, use_sampler=False, augmentation_method=None, device="cpu", use_linegraph=False):
        self.reset_g_epoch = reset_g_epoch
        self.num_workers = num_workers
        self.use_sampler = use_sampler
        self.augmentation_method = augmentation_method
        # Seems device is not useful here
        if device is not None:
            self.device = "cuda:" + str(device[0])
        else:
            self.device = "cpu"
        self.dataset = None
        self.use_linegraph = use_linegraph

    def build_dataloader(self, dataset, train_val_test=None):
        # if train_val_test in ["train", "val", "test"]:
        if False:
            num_workers = self.num_workers
            persistent_workers = True
        else:
            num_workers = 0
            persistent_workers = False
        if hasattr(dataset, "use_linegraph") and dataset.use_linegraph:
            # import ipdb; ipdb.set_trace()
            efeats = dataset.efeats
            if scipy.sparse.isspmatrix(efeats):
                efeats = efeats.A
            efeats = torch.FloatTensor(efeats)
            dataset.graph.edata["nfeat"] = efeats
            if dataset.as_directed:
                dataset.graph = get_line_graph(dataset.graph, as_directed=True)
            else:
                dataset.graph = get_line_graph(dataset.graph, as_directed=False, copy=True)
            print("Linegrpah # of edges:", dataset.graph.number_of_edges())
            dataset.graph = dataset.graph.add_self_loop()
            dataset.nfeats = dataset.graph.ndata["nfeat"]
            # dataset.graph.ndata["nfeat"] = dataset.graph.ndata["efeat"]  # In case errors
            if train_val_test == "train":
                dataset.nodes_mask = dataset.graph.ndata["train_mask"]
            elif train_val_test == "val":
                dataset.nodes_mask = dataset.graph.ndata["val_mask"]
            elif train_val_test == "test":
                dataset.nodes_mask = dataset.graph.ndata["test_mask"]
            # self.dataset = dataset

        if self.use_sampler:
            sampler = NodeAugNeighborSampler(augmentation_method=self.augmentation_method)
            dataset = dataset[0]
            dataloader = dgl.dataloading.DataLoader(
                dataset["g"],
                # Train/val/test is already splited using the line below
                torch.where(dataset["nodes_mask"])[0],
                graph_sampler=sampler,
                # device=self.device,
                batch_size=100,
                # collate_fn=partial(collate_fn, reset_g_epoch=self.reset_g_epoch),
                # use_prefetch_thread=persistent_workers,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )
        else:
            sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                collate_fn=partial(collate_fn, reset_g_epoch=self.reset_g_epoch),
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )
        return dataloader
