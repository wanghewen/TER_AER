import collections
import os
from copy import deepcopy

import datatable
import dgl
import torch
import CommonModules as CM
import numpy as np
import scipy.sparse

def get_line_graph(G: dgl.DGLGraph, as_directed=True, bidirected=True, copy=True, remove_parallel=False):
    # Used for undirected graph
    if as_directed:
        return dgl.line_graph(G, backtracking=bidirected, shared=True)
    else:  # Regard original graph as an undirected graph
        old_line_graph: dgl.DGLGraph = dgl.line_graph(G, backtracking=bidirected, shared=True)
        if copy:
            G = deepcopy(G)
        old_line_graph.ndata["parallel_group"] = -torch.ones(old_line_graph.number_of_nodes(),
                                                             dtype=torch.long).to(G.device)
        number_of_edges = G.number_of_edges()
        G.edata["index"] = torch.arange(number_of_edges, dtype=torch.long).to(G.device)
        edges_src, edges_dst = G.edges("uv")
        src_dst = torch.cat([edges_src.unsqueeze(1), edges_dst.unsqueeze(1)], dim=1)
        G.edata["parallel_group"] = torch.unique(src_dst, dim=0, return_inverse=True, sorted=False)[1]
        # Convert it to undirected graph by adding reverse edges
        G.add_edges(edges_dst, edges_src,
                     data={"new_added_edges_mask": torch.ones(edges_dst.shape[0]).to(edges_dst.device),
                           "index": torch.arange(number_of_edges,
                                                 number_of_edges + edges_dst.shape[0]).to(edges_dst.device),
                           "parallel_group": G.edata["parallel_group"]})
        new_line_graph = dgl.line_graph(G, backtracking=bidirected, shared=True)
        # The newly added reverse edge will be mapped to its corresponding original edge
        u, v = new_line_graph.edges(form="uv")
        u_data = new_line_graph.ndata["parallel_group"][u].unsqueeze(0)
        v_data = new_line_graph.ndata["parallel_group"][v].unsqueeze(0)
        u = new_line_graph.ndata["index"][u].unsqueeze(0)
        v = new_line_graph.ndata["index"][v].unsqueeze(0)
        u_v = torch.concat([u, v], dim=0)
        u_v_data = torch.concat([u_data, v_data], dim=0)
        partial_u = u_v[0][(u_v[0] >= number_of_edges) | (u_v[1] >= number_of_edges)]
        partial_v = u_v[1][(u_v[0] >= number_of_edges) | (u_v[1] >= number_of_edges)]
        partial_u_data = u_v_data[0][(u_v[0] >= number_of_edges) | (u_v[1] >= number_of_edges)]
        partial_v_data = u_v_data[1][(u_v[0] >= number_of_edges) | (u_v[1] >= number_of_edges)]
        partial_u[partial_u >= number_of_edges] -= number_of_edges
        partial_v[partial_v >= number_of_edges] -= number_of_edges
        old_line_graph.add_edges(partial_u, partial_v)
        old_line_graph.ndata["parallel_group"][partial_u] = partial_u_data
        old_line_graph.ndata["parallel_group"][partial_v] = partial_v_data
        G: dgl.DGLGraph = dgl.remove_self_loop(dgl.to_simple(old_line_graph))
        if remove_parallel:  # Remove connection between parallel edges in the original graph G
            parallel_edges = G.filter_edges(lambda edges: edges.src["parallel_group"]==edges.dst["parallel_group"])
            G.remove_edges(parallel_edges)
        return G


def get_ppr_graph(graph: dgl.DGLGraph, edge_to_node_graph: dgl.DGLGraph, experiment_data_folder,
                  subfolder, topppr_k, alpha):
    """

    :param graph: original graph
    :param experiment_data_folder: place for storing ppr results
    :param subfolder: place for storing ppr results
    :param topppr_k: number of edges to be considered
    :return:
    """
    # eid_connection_limit = 10
    if alpha is not None:
        subfolder += f"_alpha={alpha}"
    experiment_data_subfolder = os.path.join(experiment_data_folder,
                                            "topppr_original/dataset", subfolder)
    topkid_path = os.path.join(experiment_data_subfolder, "topkid")
    topkprob_path = os.path.join(experiment_data_subfolder, "topkprob")
    # We make sure the top ids in the edge to node graph are for transaction nodes (E-Nodes)
    CM.Utilities.TimeElapsed()
    with open(topkid_path, "r+") as fd_id, open(topkprob_path, "r+") as fd_prob:
        # import pdb;pdb.set_trace()
        ids = fd_id.read()
        ids = datatable.fread(ids, sep=" ", header=False)
        ids = ids.to_numpy(stype="int32")

        k = min(topppr_k, ids.shape[1])
        eid_connection_limit = k

        ids = ids[:, :k]
        ids = ids.reshape(-1)
        # ids = ids[:, :k].reshape(-1)
        print("Load ids, time elapsed:", CM.Utilities.TimeElapsed(LastTime=True))
        probs = fd_prob.read()
        probs = datatable.fread(probs, sep=" ", header=False)
        probs = probs.to_numpy(stype="float32")
        probs = probs[:, :k]
        probs = probs.reshape(-1)
        # probs = temp_probs
        # probs = probs[:, :k].reshape(-1)
        print("Load probs, time elapsed:", CM.Utilities.TimeElapsed(LastTime=True))
        node_idxs = np.arange(edge_to_node_graph.number_of_nodes(), dtype=np.int32)
        node_idxs = np.repeat(node_idxs, k)
        # import ipdb; ipdb.set_trace()
        topk_ppr_matrix = scipy.sparse.csr_matrix((probs, (node_idxs, ids)), dtype=np.float32,
                                                  shape=(edge_to_node_graph.number_of_nodes(),
                                                         edge_to_node_graph.number_of_nodes()))

    # mask = (ids < edge_to_node_graph.number_of_nodes()) & (node_idxs < edge_to_node_graph.number_of_nodes())
    # ids = ids[mask]
    # node_idxs = node_idxs[mask]
    # ids = ids.tolist()
    # node_idxs = node_idxs.tolist()
    topk_ppr_matrix: scipy.sparse.csr_matrix = topk_ppr_matrix[:graph.number_of_edges(), :graph.number_of_edges()]
    node_topk_ppr_eids = collections.defaultdict(dict)
    for eid, eid_2 in zip(*topk_ppr_matrix.nonzero()):
        node_topk_ppr_eids[eid][eid_2] = 1
        node_topk_ppr_eids[eid_2][eid] = 1
    us, vs, eids = graph.edges("all")
    us, vs, eids = us.tolist(), vs.tolist(), eids.tolist()
    node_src_neighbor_eids = collections.defaultdict(dict)
    node_dst_neighbor_eids = collections.defaultdict(dict)
    parallel_eids_uv = collections.defaultdict(dict)
    for u, v, eid in zip(us, vs, eids):
        node_src_neighbor_eids[u][eid] = 1
        node_dst_neighbor_eids[v][eid] = 1
        parallel_eids_uv[eid] = (u, v)

    graph_2 = deepcopy(edge_to_node_graph)
    graph_2.remove_edges(torch.arange(graph_2.number_of_edges()))
    # Only keep E-Nodes
    graph_2.remove_nodes(torch.arange(graph.number_of_edges(), graph_2.number_of_nodes()))
    add_edges_src = []
    add_edges_dst = []
    eid_connection_count = collections.defaultdict(int)
    for u, v, eid in zip(us, vs, eids):
        for eid_2 in node_topk_ppr_eids[eid]:
            if parallel_eids_uv[eid] == parallel_eids_uv[eid_2]:
                continue
            if eid_2 in node_src_neighbor_eids[u] and eid != eid_2:
                eid_connection_count[eid] += 1
                if eid_connection_count[eid] > eid_connection_limit:
                    break
                add_edges_src.append(eid)
                add_edges_dst.append(eid_2)
                add_edges_src.append(eid_2)
                add_edges_dst.append(eid)
    graph_2.add_edges(add_edges_src, add_edges_dst)
    graph_2 = dgl.to_simple(graph_2)

    graph_3 = deepcopy(edge_to_node_graph)
    graph_3.remove_edges(torch.arange(graph_3.number_of_edges()))
    # Only keep E-Nodes
    graph_3.remove_nodes(torch.arange(graph.number_of_edges(), graph_3.number_of_nodes()))
    add_edges_src = []
    add_edges_dst = []
    eid_connection_count = collections.defaultdict(int)
    for u, v, eid in zip(us, vs, eids):
        for eid_2 in node_topk_ppr_eids[eid]:
            if parallel_eids_uv[eid] == parallel_eids_uv[eid_2]:
                continue
            if eid_2 in node_dst_neighbor_eids[v] and eid != eid_2:
                eid_connection_count[eid] += 1
                if eid_connection_count[eid] > eid_connection_limit:
                    break
                add_edges_src.append(eid)
                add_edges_dst.append(eid_2)
                add_edges_src.append(eid_2)
                add_edges_dst.append(eid)
    graph_3.add_edges(add_edges_src, add_edges_dst)
    graph_3 = dgl.to_simple(graph_3)

    return graph_2, graph_3
    # # Example:
    # G=dgl.graph(([0,1,1,3],[1,2,3,4]))
    # LG=get_line_graph(G, as_directed=False, copy=True)
    # LGA=LG.adjacency_matrix(scipy_fmt="csr")
    # topk_ppr_matrix=LGA
    # graph=G
    # us, vs, eids = graph.edges("all")
    # us, vs, eids = us.tolist(), vs.tolist(), eids.tolist()
    # node_src_neighbor_eids = collections.defaultdict(dict)
    # node_dst_neighbor_eids = collections.defaultdict(dict)
    # for u, v, eid in zip(us, vs, eids):
    #     node_src_neighbor_eids[u][eid] = 1
    #     node_dst_neighbor_eids[v][eid] = 1
    # add_edges_src = []
    # add_edges_dst = []
    # for u, v, eid in zip(us, vs, eids):
    #     src_eids = set(topk_ppr_matrix[:, eid].nonzero()[0].tolist())
    #     dst_eids = set(topk_ppr_matrix[eid].nonzero()[1].tolist())
    #     for eid_2 in src_eids | dst_eids:
    #         if eid_2 in node_src_neighbor_eids[u] and eid != eid_2:
    #             add_edges_src.append(eid)
    #             add_edges_dst.append(eid_2)
    #             add_edges_src.append(eid_2)
    #             add_edges_dst.append(eid)
    # add_edges_src: [1, 2, 2, 1]
    # add_edges_dst: [2, 1, 1, 2]


def permute_sparse_matrix(M, new_row_order=None, new_col_order=None):
    """
    https://stackoverflow.com/a/63058622/4656538
    Reorders the rows and/or columns in a scipy sparse matrix
        using the specified array(s) of indexes
        e.g., [1,0,2,3,...] would swap the first and second row/col.
    """
    if new_row_order is None and new_col_order is None:
        return M

    new_M = M
    if new_row_order is not None:
        I = scipy.sparse.eye(M.shape[0]).tocoo()
        I.row = I.row[new_row_order]
        new_M = I.dot(new_M)
    if new_col_order is not None:
        I = scipy.sparse.eye(M.shape[1]).tocoo()
        I.col = I.col[new_col_order]
        new_M = new_M.dot(I)
    return new_M


def zero_rows(M, rows_to_zero):

    ixs = np.ones(M.shape[0], int)
    ixs[rows_to_zero] = 0
    D = scipy.sparse.diags(ixs)
    res = D @ M
    return res
