import datetime
import os
import subprocess
import sys

import dgl
import math
import scipy.linalg
import scipy.sparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.linalg import clarkson_woodruff_transform
# from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score, \
    average_precision_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import normalize
from torch import nn
import scipy.sparse as sp
import torch
import numpy as np
# from xgboost import XGBClassifier
import pytorch_lightning as pl

from utils import import_numpy_array, export_numpy_array

try:
    from sparse_dot_mkl import dot_product_mkl
except ImportError:
    pass


def dot_product(x, y, dense=False, **kwargs):
    if x.dtype == "float64":
        x = x.astype("float32", copy=False)
    if y.dtype == "float64":
        y = y.astype("float32", copy=False)
    # result = dot_product_mkl(x, y, cast=True, dense=dense)
    # return result
    z = x @ y
    return z.A if dense and scipy.sparse.issparse(z) else z

# from analysis.matrix_error import epp_vs_lra
from dataloaders.FC_dataloader import FCDataloader
from icecream import ic
import CommonModules as CM

from models.Base import BaseModel
from models.FC import FCBaseline
from models.functions.svd import randomized_svd

USED_CLASS = BaseModel


def convert_matrix_to_numpy(m):
    if isinstance(m, scipy.sparse.spmatrix):
        m = m.A
    elif isinstance(m, np.matrix):
        m = m.A
    elif isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    return m


def evaluate(labels, scores, multilabel=False):
#     scores = scores.cpu()
#     labels = labels.cpu()
    if multilabel:
        label_column_mask = np.array(labels.sum(0) > 1).reshape(-1)  # Exclude class with too few samples
        labels = labels[:, label_column_mask]
        scores = scores[:, label_column_mask]
        average_precision = average_precision_score(labels, scores, average="macro")
        roc_auc = roc_auc_score(labels, scores, average="macro")
        macro_f1 = f1_score(labels, scores > 0.5, average="macro")
        return None, roc_auc, average_precision, macro_f1, int(labels.sum())
    else:
        predictions_test = np.array(scores)
        precision, recall, thresholds = precision_recall_curve(labels, predictions_test)
        f1_scores = 2 * recall * precision / (recall + precision)
        threshold = thresholds[np.nanargmax(f1_scores)]
        best_anomaly_f1 = np.nanmax(f1_scores)
        best_macro_f1 = f1_score(labels, predictions_test>threshold, average="macro")
        roc_auc = roc_auc_score(labels, scores, average="weighted")
        average_precision = average_precision_score(labels, scores)
        print("Threshold:", threshold)
        print(classification_report(labels, predictions_test>threshold))
        return best_anomaly_f1, roc_auc, average_precision, best_macro_f1, int(labels.sum())


class GNNVanilla(USED_CLASS):
    def __init__(self,
                 #                  g,
                 in_nfeats,
                 in_efeats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 structure,
                 activation=torch.nn.ReLU(),
                 rel_names=None,
                 target_node_type=None,
                 pos_weight=None,
                 use_smote=False,
                 random_adj=False,
                 # Either False, "once"(random at first epoch) or "every_epoch" or "every_epoch_double_norm"
                 adj_method=None,
                 feature_augment=False,
                 use_gb=False,
                 edge_attributed=False,  # Whether use edge attributed graph or edge to node graph
                 svd_dim=256,  # None means not use SVD to reduce feature dim first
                 use_random_feature=False,  # Add an extra 1-dim column as random feature
                 train_ratio=1.0,  # Reduce training data to save training time for large dataset
                 use_epp=False,
                 use_ter=False,
                 use_lra=False,
                 use_node2vec=False,
                 use_edge2vec=False,
                 use_aer=False,
                 use_ppr=False,
                 use_hkpr=False,
                 ppr_decay_factor=0.5,
                 hkpr_heat_constant=1,
                 error_threshold=0.00001,
                 error_analysis=False,
                 classifier="logistic",
                 cache_folder_path=None,
                 **kwargs):
        super(USED_CLASS, self).__init__()
        #         self.g = g
        self.n_classes = n_classes
        self.dropout = dropout
        self.rel_names = rel_names
        self.target_node_type = target_node_type
        self.augment_method = None
        self.use_smote = use_smote
        assert random_adj in [False, "once", "every_epoch", "every_epoch_double_norm"]
        self.random_adj = random_adj
        self.randomed_adj_flag = False
        self.adj_method = adj_method
        self.feature_augment = feature_augment
        self.augmented_data = None
        self.smote_adj = None
        self.use_gb = use_gb
        self.edge_attributed = edge_attributed
        self.svd_dim = svd_dim
        self.use_random_feature = use_random_feature
        self.edge_features = None
        self.use_epp = use_epp
        self.use_ter = use_ter
        self.use_lra = use_lra
        self.use_node2vec = use_node2vec
        self.use_edge2vec = use_edge2vec
        self.use_aer = use_aer
        self.use_ppr = use_ppr
        self.use_hkpr = use_hkpr
        self.ppr_decay_factor = ppr_decay_factor
        self.hkpr_heat_constant = hkpr_heat_constant
        self.error_threshold = error_threshold
        self.error_analysis = error_analysis
        assert classifier in ["logistic", "mlp"]
        self.classifier = classifier
        self.cache_folder_path = cache_folder_path
        if self.use_gb:
            self.gb_model_trained = False
            # self.gb_model = GradientBoost()
            self.gb_model = LogisticRegression(verbose=1, max_iter=1000)
            in_nfeats += 1
        self.norm_edge_weight = nn.Parameter(torch.ones(1))  # Dummy parameter to be modified in forward
        self.weights = nn.ParameterList()
        self.bns = nn.ModuleList()
        self.normalized_adj = None
        self.train_ratio = train_ratio
        self._mask = None
        #         self.last_layer = nn.Linear(n_hidden*2+in_efeats, n_classes) # Use this one to concat edge attrs and get labels
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if structure in ["GCN", "SGC"]:
            # input layer
            # self.layers.append(GraphConv(in_nfeats, n_hidden, activation=activation))
            self.weights.append(nn.Parameter(torch.empty(in_nfeats, n_hidden)))
            self.bns.append(nn.BatchNorm1d(n_hidden))
            # hidden layers
            for i in range(n_layers - 1):
                # self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
                self.weights.append(nn.Parameter(torch.empty(n_hidden, n_hidden)))
                self.bns.append(nn.BatchNorm1d(n_hidden))
                # self.layers.append(nn.Dropout(p=dropout))
        # else:
        #     raise NotImplementedError
        for weight in self.weights:
            torch.nn.init.xavier_uniform_(weight)
        # self.last_layer = torch.nn.Linear(n_hidden, n_classes)
        self.last_layer_linear = torch.nn.Linear(n_hidden, n_classes)
        self.last_layer = torch.nn.Sequential(torch.nn.BatchNorm1d(n_hidden), self.last_layer_linear)
        ic(self.weights)

    @staticmethod
    def _normalize(mx, norm="row"):
        """Normalize sparse matrix. For both it will use out degree (Not a problem for symmetric adj)"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        if norm == "row":
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
        elif norm == "both":
            r_inv_sqrt = np.sqrt(r_inv)
            r_mat_inv = sp.diags(r_inv_sqrt)
            mx = r_mat_inv @ mx @ r_mat_inv
        else:
            raise ValueError("Norm is not correct. Should be either 'row' or 'both'")
        return mx

    @staticmethod
    def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    @staticmethod
    def _torch_sparse_tensor_to_sparse_mx(sparse_tensor: torch.sparse.FloatTensor):
        CoalesceTensor = sparse_tensor.cpu().coalesce()
        TensorIndices = CoalesceTensor.indices().numpy()
        TensorValues = CoalesceTensor.values().numpy()
        out = sp.coo_matrix((TensorValues, (TensorIndices[0], TensorIndices[1])), shape=sparse_tensor.shape,
                                      dtype=TensorValues.dtype)
        return out

    def _ter(self, incidence_matrix, rep_dimension, ppr_decay_factor=None, hkpr_heat_constant=None,
             cache_folder_path=None):
        """

        :param incidence_matrix: Transpose of the incidence matrix
        :param alpha:
        :param rep_dimension:
        :return:
        """
        # random_matrix = np.random.normal(size=(incidence_matrix.shape[0], rep_dimension))  # m*k
        print("_ter started...")
        cache_u_path = os.path.join(str(cache_folder_path), "ter_cache_u.blp")
        cache_sigma_path = os.path.join(str(cache_folder_path), "ter_cache_sigma.blp")
        if cache_folder_path is not None and CM.IO.FileExist(cache_u_path):
            print("load from _ter cache files from:", cache_folder_path)
            u = import_numpy_array(cache_u_path)
            sigma = import_numpy_array(cache_sigma_path)
        else:
            incidence_matrix = incidence_matrix.tocsr()
            i_m = dot_product(sp.diags(1 / incidence_matrix.sum(1).A.flatten(), format="csr"), incidence_matrix)  # F, m*k
            i_n = dot_product(incidence_matrix, sp.diags(1 / incidence_matrix.sum(0).A.flatten(), format="csr"))  # m*k
            i_n_T = i_n.T.tocsr()  # Better efficiency, B, k*m
            # sample_matrix = i_m @ (i_n_T @ random_matrix)  # m*k
            # q_l, q_r = scipy.linalg.qr(sample_matrix, mode='economic')  # m*k, k*k
            # svd_matrix = (q_l.T @ i_m) @ i_n_T # k*m
            # u, sigma, vT = scipy.linalg.svd(svd_matrix, full_matrices=False)  # k*k, k, k*k
            u, sigma, vT = randomized_svd(i_m, i_n_T, rep_dimension, random_state=12)  # m*k, k, k*k
            # u, sigma, vT = scipy.linalg.svd(svd_matrix, full_matrices=True)  # k*m, m, m*k
            # u = q_l @ u  # m*k
            if cache_folder_path is not None:
                print("export to _ter cache files at:", cache_folder_path)
                os.makedirs(cache_folder_path, exist_ok=True)
                export_numpy_array(u, cache_u_path)
                export_numpy_array(sigma, cache_sigma_path)
        if ppr_decay_factor:
            sigma = (1-(1-ppr_decay_factor)*sigma)/ppr_decay_factor
            sigma = np.diag(np.sqrt(1 / sigma))  # k*k
            z = u @ sigma
        else:
            sigma = np.sqrt(np.exp(hkpr_heat_constant * sigma) / np.exp(hkpr_heat_constant))
            sigma = np.diag(sigma)  # k*k
            z = u @ sigma
        return z


    def _epp_vanilla(self, edge_features, incidence_matrix, predicted_labels=None):
        """

        :param edge_features:
        :param incidence_matrix: Transpose of the incidence matrix
        :param predicted_labels:
        :return:
        """
        ################ Undirected edge to edge ####################
        # i_m = i_n = sp.diags(np.sqrt(1 / incidence_matrix.sum(1).A.flatten())) @ incidence_matrix @ \
        # sp.diags(np.sqrt(1 / incidence_matrix.sum(0).A.flatten()))
        ################ Undirected edge to edge ######################
        i_m = sp.diags(1 / incidence_matrix.sum(1).A.flatten()) @ incidence_matrix
        i_n = incidence_matrix @ sp.diags(1 / incidence_matrix.sum(0).A.flatten())
        i_n_T = i_n.T.tocsr()  # Better efficiency
        ################ Directed edge to edge ######################
        # incidence_matrix_1 = g.incidence_matrix("in")
        # incidence_matrix_2 = g.incidence_matrix("out")
        # incidence_matrix_1 = self._torch_sparse_tensor_to_sparse_mx(incidence_matrix_1).T
        # incidence_matrix_2 = self._torch_sparse_tensor_to_sparse_mx(incidence_matrix_2).T
        #
        # i_m = sp.diags(1 / incidence_matrix_1.sum(1).A.flatten()) @ incidence_matrix_1
        # i_n = incidence_matrix_2 @ sp.diags(1 / incidence_matrix_2.sum(0).A.flatten())
        ##########################################################
        CM.Utilities.TimeElapsed(LastTime=True)
        print("_epp_vanilla started...")
        if self.svd_dim and edge_features.shape[1] > self.svd_dim:
            # print("Start doing SVDS... SVD dim:", self.svd_dim)
            # edge_features = normalize(edge_features, norm='l2', axis=1)
            # # u, s, vT = svds(edge_features, k=self.svd_dim)
            # u, s, vT = svds(edge_features, k=min(self.svd_dim, edge_features.shape[1]-1))
            # # u, s, vT = svds(edge_features, k=min(5, edge_features.shape[1]-1))
            # print("Singular values:", s, "used time:", CM.Utilities.TimeElapsed(LastTime=True))
            # edge_features = u
            print("Start doing Clarkson_woodruff_transform... reduced dim:", self.svd_dim)
            edge_features_T = clarkson_woodruff_transform(edge_features.T, sketch_size=self.svd_dim)
            edge_features = edge_features_T.T

        if predicted_labels is not None:
            edge_features = scipy.sparse.hstack([scipy.sparse.csr_matrix(edge_features), predicted_labels], format="csr")
        X_sum_1 = edge_features  # Pick all edges
        for iteration in range(10):
            # X_sum = 0.5 * incidence_matrix @ (incidence_matrix.T @ X_sum) + edge_features
            X_sum_1 = (1 - self.ppr_decay_factor) * i_m @ (i_n_T @ X_sum_1) + edge_features
            # X_sum_1 = 0.5 * dot_product_mkl(i_m, i_n_T @ X_sum_1) + edge_features
            print("_epp_vanilla iteration:", iteration, "used time:", CM.Utilities.TimeElapsed(LastTime=True))
        edge_features_1 = self.ppr_decay_factor * X_sum_1
        # X_sum_2 = edge_features  # Pick all edges
        # for _ in range(10):
        #     # X_sum = 0.5 * incidence_matrix @ (incidence_matrix.T @ X_sum) + edge_features
        #     X_sum_2 = 0.5 * i_n @ (i_m.T @ X_sum_2) + edge_features
        # edge_features_2 = 0.5 * X_sum_2
        # # edge_features_2 = edge_features_1
        # edge_features = edge_features_1 + edge_features_2
        if not isinstance(edge_features_1, np.ndarray):
            edge_features = edge_features_1.A
        else:
            edge_features = edge_features_1
        ##########################################
        return edge_features

    def _aer(self, edge_features, incidence_matrix, predicted_labels=None, method=None, ppr_decay_factor=None,
             hkpr_heat_constant=None):
        """

        :param edge_features:
        :param incidence_matrix: Transpose of the incidence matrix
        :param predicted_labels:
        :return:
        """
        assert method in ["ppr", "hkpr"]
        i_m = sp.diags(1 / incidence_matrix.sum(1).A.flatten(), dtype="float32") @ incidence_matrix
        i_m = i_m.tocsc()
        i_n = incidence_matrix @ sp.diags(1 / incidence_matrix.sum(0).A.flatten())
        # i_n_T = i_n.T.tocsr()  # Better efficiency
        i_n_T = i_n.T.tocsc()
        CM.Utilities.TimeElapsed(LastTime=True)
        print(f"_aer using {method} started...")
        # if self.svd_dim and edge_features.shape[1] > self.svd_dim:
        #     print("Start doing Clarkson_woodruff_transform... reduced dim:", self.svd_dim)
        #     edge_features_T = clarkson_woodruff_transform(edge_features.T, sketch_size=self.svd_dim)
        #     edge_features = edge_features_T.T

        if predicted_labels is not None:
            edge_features = scipy.sparse.hstack([scipy.sparse.csr_matrix(edge_features), predicted_labels], format="csr")
        if isinstance(edge_features, scipy.sparse.csr_matrix):
            # T = edge_features.tocsc()  # Pick all edges
            T = edge_features.A  # Pick all edges
        else:
            T = edge_features
        if method == "ppr":
            X = ppr_decay_factor * T
        elif method == "hkpr":
            X = math.exp(-hkpr_heat_constant) * T
        # import ipdb; ipdb.set_trace()
        X_j_inverse_diag = scipy.sparse.diags(1/convert_matrix_to_numpy(X.sum(0)).reshape(-1),
                                              shape=(X.shape[1], X.shape[1]),
                                              format="csr")
        X_j_mask = np.ones(X_j_inverse_diag.shape[1], dtype=bool)
        if method == "ppr":
            max_iteration = int(math.log(1 / self.error_threshold, 1 / (1 - ppr_decay_factor)) - 1) + 1
        elif method == "hkpr":
            poisson_summation = 0
            for max_iteration in range(20):
                poisson_summation += math.exp(-hkpr_heat_constant) * (hkpr_heat_constant ** max_iteration) /\
                                     math.factorial(max_iteration)
                if 1 - poisson_summation < self.error_threshold:
                    break
        for iteration in range(max_iteration):
            # print(type(i_m), type(i_n_T), type(T))
            # print(type(i_n_T @ T[:, X_j_mask]), type(i_m))
            # print(type(i_m @ (i_n_T @ T[:, X_j_mask])))
            # print(type(T[:, X_j_mask]))
            X_j_mask_matrix = scipy.sparse.diags(X_j_mask,
                                                 shape=(X_j_mask.shape[0], X_j_mask.shape[0]),
                                                 format="csc", dtype=float)  # Use Diagonal matrix to select columns
            # import ipdb; ipdb.set_trace()
            # from IPython import embed
            # embed()
            # Replace data for those columns in X_j_mask_matrix. But for now, the feature matrix will be dense.
            # T = T - dot_product(T, X_j_mask_matrix) + \
            #     dot_product(i_m, dot_product(i_n_T, dot_product(T, X_j_mask_matrix)))
            # T[:, X_j_mask] = i_m @ (i_n_T @ T[:, X_j_mask])
            T[:, X_j_mask] = dot_product(i_m, dot_product(i_n_T, T[:, X_j_mask]))
            if method == "ppr":
                # X = X + ppr_decay_factor * \
                #     ((1 - ppr_decay_factor) ** (iteration + 1)) * dot_product(T, X_j_mask_matrix)
                X[:, X_j_mask] = X[:, X_j_mask] + ppr_decay_factor * \
                                 ((1 - ppr_decay_factor) ** (iteration + 1)) * T[:, X_j_mask]
                new_X_j_submask = ((1 - ppr_decay_factor) ** (iteration + 1) *
                                   dot_product(T[:, X_j_mask], X_j_inverse_diag[X_j_mask][:, X_j_mask])).max(0) > \
                                  self.error_threshold
            elif method == "hkpr":
                # X = X + math.exp(-hkpr_heat_constant) * \
                #     (hkpr_heat_constant ** (iteration + 1) /
                #      math.factorial(iteration + 1)) * dot_product(T, X_j_mask_matrix)
                X[:, X_j_mask] = X[:, X_j_mask] + math.exp(-hkpr_heat_constant) * \
                                 (hkpr_heat_constant ** (iteration + 1) / math.factorial(iteration + 1)) * T[:, X_j_mask]
                new_X_j_submask = ((hkpr_heat_constant ** (iteration + 1) / math.factorial(iteration + 1)) *
                                   dot_product(T[:, X_j_mask], X_j_inverse_diag[X_j_mask][:, X_j_mask])).max(0) > \
                                  self.error_threshold
            # import ipdb; ipdb.set_trace()
            new_X_j_submask = convert_matrix_to_numpy(new_X_j_submask).reshape(-1)
            X_j_mask[X_j_mask] = new_X_j_submask
            # X_sum_1 = 0.5 * dot_product_mkl(i_m, i_n_T @ X_sum_1) + edge_features
            print("_aer iteration:", iteration, "current attribute set size:", X_j_mask.sum(), "used time:",
                  CM.Utilities.TimeElapsed(LastTime=True))
        edge_features_1 = X
        # X_sum_2 = edge_features  # Pick all edges
        # for _ in range(10):
        #     # X_sum = 0.5 * incidence_matrix @ (incidence_matrix.T @ X_sum) + edge_features
        #     X_sum_2 = 0.5 * i_n @ (i_m.T @ X_sum_2) + edge_features
        # edge_features_2 = 0.5 * X_sum_2
        # # edge_features_2 = edge_features_1
        # edge_features = edge_features_1 + edge_features_2
        edge_features = convert_matrix_to_numpy(edge_features_1)
        # if not isinstance(edge_features_1, np.ndarray):
        #     edge_features = edge_features_1.A
        # else:
        #     edge_features = edge_features_1
        ##########################################
        return edge_features

    def _node2vec(self, adj: scipy.sparse.csr_matrix):
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        adj = adj.tocoo()
        edgelist = np.concatenate([adj.row.reshape(-1, 1), adj.col.reshape(-1, 1)], axis=1)
        output_folder = "./node2vec_data"
        os.makedirs(output_folder, exist_ok=True)
        output_edgelist_path = os.path.join(output_folder, "data.edgelist")
        output_embedding_path = os.path.join(output_folder, "data.emb")
        np.savetxt(output_edgelist_path, edgelist, fmt="%d")
        subprocess.run(["./snap/examples/node2vec/node2vec", f"-i:{output_edgelist_path}",
                        f"-o:{output_embedding_path}", "-dr", "-v"])
        embeds = np.loadtxt(output_embedding_path, skiprows=1)
        # import ipdb; ipdb.set_trace()
        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        print(f"_node2vec time elapsed: {end_training_time - start_training_time} sec")
        return embeds

    def _edge2vec(self, g: dgl.DGLGraph):
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        adj = g.adj()
        u, v = g.edges()
        u += 1
        v += 1
        output_folder = "./models/Edge2vec"
        output_edgelist_path = os.path.join(output_folder, "current.graph")
        output_embedding_path = os.path.join(output_folder, "results/embedding.log")
        with open(output_edgelist_path, "w") as fd:
            for u, v in zip(u.tolist(), v.tolist()):
                print(u, v, file=fd)
        subprocess.run(["python", "edge2vec.py", f"-i", "current.graph",
                        f"-m", "results", "-n", f"{adj.shape[0]}", "-s", "500"], cwd=output_folder)
        embeds = np.loadtxt(output_embedding_path)
        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        print(f"_edge2vec time elapsed: {end_training_time - start_training_time} sec")
        return embeds


    def fit(self, dataloaders=None, max_epochs=1, gpus=None, **kwargs):
        """
        Use Sklearn instead of Pytorch-lightning to fit the data

        :param dataloaders:
        :param max_epochs:
        :param gpus:
        :return:
        """
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        train_data = next(iter(train_dataloader))
        val_data = next(iter(val_dataloader))
        test_data = next(iter(test_dataloader))
        if self.edge_attributed:  # Use edge attributed graph (Not edge to node graph)
            mask_key = "edges_mask"
            feature_key = "efeats"
        else:
            mask_key = "nodes_mask"
            feature_key = "nfeats"
        if isinstance(train_data[feature_key], torch.Tensor):
            train_mask = train_data[mask_key]
            val_mask = val_data[mask_key]
            test_mask = test_data[mask_key]
            labels = train_data["labels"].cpu().numpy()
        else:
            train_mask = train_data[mask_key].cpu().numpy()
            val_mask = val_data[mask_key].cpu().numpy()
            test_mask = test_data[mask_key].cpu().numpy()
            labels = train_data["labels"]
        # print("label distribution:", labels[train_mask | val_mask | test_mask].sum(0))
        # print("no label edges:", (labels[train_mask | val_mask | test_mask].sum(1)==0).sum())
        # return
        # import os
        # train_embeds, train_labels = CM.IO.ImportFromPkl(
        #     os.path.join("/data/PublicGraph/DBLP/data_count_vectorizer", "train_embeds_labels.pkl"))
        # test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join("/data/PublicGraph/DBLP/data_count_vectorizer_split_by_node", "test_embeds_labels.pkl"))
        if self.train_ratio < 1:
            train_mask[np.where(train_mask)[0][int(train_mask.sum()*self.train_ratio):]] = False

        if self.edge_attributed:  # Use edge attributed graph (Not edge to node graph)
            if isinstance(train_data[feature_key], torch.Tensor):
                edge_features = train_data[feature_key][train_data["edges_id"]].cpu().numpy().astype(np.float32)
            else:
                edge_features = train_data[feature_key][train_data["edges_id"]].astype(np.float32)
            if isinstance(edge_features, scipy.sparse.csr_matrix):
                print("nnz(X):", edge_features.nnz)
            else:
                print("nnz(X):", np.count_nonzero(edge_features))
            # return
            if self.use_random_feature:
                edge_features = scipy.sparse.hstack([edge_features, scipy.sparse.rand(edge_features.shape[0], 1, density=1.0)], format="csr")
            predicted_labels = None
            if self.use_gb:
                self.gb_model.fit(edge_features[train_mask], labels[train_mask])
                self.gb_model_trained = True
                predicted_labels = self.gb_model.predict_proba(edge_features)[:, [1]]
            if self.use_epp or self.use_ter or self.use_lra or self.use_aer or self.use_aer:
                ##############Do Propagation##############
                incidence_matrix = train_data["g"].incidence_matrix("in") + train_data["g"].incidence_matrix("out")
                incidence_matrix = self._torch_sparse_tensor_to_sparse_mx(incidence_matrix).T
                # features_ter = edge_features[:, :100]  # For debug purpose
                features = []
                if self.use_ter or self.use_lra:
                    start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    if self.use_ppr:
                        features_ter = self._ter(incidence_matrix, self.svd_dim or edge_features.shape[1],
                                                 ppr_decay_factor=self.ppr_decay_factor,
                                                 cache_folder_path=self.cache_folder_path)
                    elif self.use_hkpr:
                        features_ter = self._ter(incidence_matrix, self.svd_dim or edge_features.shape[1],
                                                 hkpr_heat_constant=self.hkpr_heat_constant,
                                                 cache_folder_path=self.cache_folder_path)
                    else:
                        features_ter = self._ter(incidence_matrix, self.svd_dim or edge_features.shape[1],
                                                 ppr_decay_factor=0.5,
                                                 cache_folder_path=self.cache_folder_path)
                    end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    print(f"_ter time elapsed: {end_training_time - start_training_time} sec")
                    if self.use_ter:
                        features.append(features_ter)
                assert (not all([self.use_epp, self.use_lra])) or self.error_analysis  # Use only one of them
                if self.use_lra:
                    start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    # Potential problems as we don't do dimension reduction here, compared with self._epp_vanilla
                    # But we do it during preprocessing steps, so normally it should be fine
                    features_lra = features_ter @ (features_ter.T @ edge_features)
                    # if self.error_analysis:
                    #     print("Starting error analysis...")
                    #     features_epp = self._epp_vanilla(edge_features, incidence_matrix, predicted_labels)
                    #     epp_vs_lra(features_epp, features_lra)
                    #     return
                    end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    print(f"_lra time elapsed: {end_training_time - start_training_time} sec")
                    features.append(features_lra)
                elif self.use_epp:
                    start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    features_epp = self._epp_vanilla(edge_features, incidence_matrix, predicted_labels)
                    end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    print(f"_epp_vanilla time elapsed: {end_training_time - start_training_time} sec")
                    features.append(features_epp)
                elif self.use_aer and self.use_ppr:
                    start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    features_epp = self._aer(edge_features, incidence_matrix, predicted_labels,
                                             method="ppr", ppr_decay_factor=self.ppr_decay_factor)
                    end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    print(f"_aer_ppr time elapsed: {end_training_time - start_training_time} sec")
                    features.append(features_epp)
                elif self.use_aer and self.use_hkpr:
                    start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    # import warnings
                    # import traceback
                    # warnings.filterwarnings("error")  # Treat warnings as errors
                    # try:
                    features_epp = self._aer(edge_features, incidence_matrix, predicted_labels,
                                             method="hkpr", hkpr_heat_constant=self.hkpr_heat_constant)
                    # except Warning:
                    #     print(traceback.format_exc())  # print traceback
                    #     warnings.filterwarnings("ignore")  # ignore warnings
                    #     return
                    end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                    print(f"_aer_hkpr time elapsed: {end_training_time - start_training_time} sec")
                    features.append(features_epp)
                # features = features_ter
                # features = features_epp
                sparse_flag = False
                for feature in features:
                    if scipy.sparse.isspmatrix(feature):
                        sparse_flag = True
                if sparse_flag:
                    features = scipy.sparse.hstack(features, format="csr")
                else:
                    features = np.concatenate(features, axis=1)
                # features = normalize(np.concatenate([features_ter, features_epp], axis=1), norm='l2', axis=1)
                ##########################################
            elif self.use_edge2vec:
                features = self._edge2vec(train_data["g"])
            else:
                ###########Use Raw edge features##########
                print("Using Raw Edge features")
                if self.svd_dim and self.svd_dim < edge_features.shape[1]:
                    # print("Start doing SVDS... SVD dim:", self.svd_dim)
                    # edge_features = normalize(edge_features, norm='l2', axis=1)
                    # u, s, vT = svds(edge_features, k=min(self.svd_dim, edge_features.shape[0]))
                    # print("Singular values:", s, "used time:", CM.Utilities.TimeElapsed(LastTime=True))
                    # edge_features = u
                    print("Start doing Clarkson_woodruff_transform... reduced dim:", self.svd_dim)
                    edge_features_T = clarkson_woodruff_transform(edge_features.T, sketch_size=self.svd_dim)
                    edge_features = edge_features_T.T
                if self.use_gb:
                    edge_features = np.concatenate([edge_features, predicted_labels], axis=1)
                features = edge_features
            ##########################################
        else:  # Use edge to node graph. Then similar as what SGC does
            adj = train_data["g"].adj(scipy_fmt="csr", transpose=False).astype(np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            h = train_data[feature_key]
            if self.use_node2vec:
                embeds = self._node2vec(adj)
                features = scipy.sparse.hstack([h, scipy.sparse.csr_matrix(embeds)], format="csr")
            else:
                # self.normalized_adj = self._normalize(adj + 1 * sp.eye(adj.shape[0]))
                self.normalized_adj = self._normalize(adj)
                if self.use_random_feature:
                    h = scipy.sparse.hstack([h, scipy.sparse.rand(h.shape[0], 1, density=1.0)], format="csr")
                for i, _ in enumerate(self.weights):
                    h = self.normalized_adj @ h
                features = h
                # labels = train_data["labels"].cpu().numpy()

        if self.classifier == "logistic":
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                if isinstance(features, np.ndarray) and features.shape[0] < 1e6:
                    n_jobs = 25
                else:
                    n_jobs = None
                clf = MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000, verbose=1), n_jobs=n_jobs)
            else:
                clf = LogisticRegression(verbose=1, max_iter=1000)
            print("start training classifier...")
            clf = clf.fit(features[train_mask], labels[train_mask])
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                print("train anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                      evaluate(labels[train_mask],
                               np.array(clf.predict_proba(features[train_mask]))[:, :, 1].T,
                               multilabel=True))
                print("val anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                      evaluate(labels[val_mask],
                               np.array(clf.predict_proba(features[val_mask]))[:, :, 1].T,
                               multilabel=True))
                print("test anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                      test_results := evaluate(labels[test_mask],
                                               np.array(clf.predict_proba(features[test_mask]))[:, :, 1].T,
                                               multilabel=True))
            else:
                print("train anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                      evaluate(labels[train_mask],
                               clf.predict_proba(features[train_mask])[:, 1],
                               multilabel=False))
                print("val anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                      evaluate(labels[val_mask],
                               clf.predict_proba(features[val_mask])[:, 1],
                               multilabel=False))
                print("test anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                      test_results := evaluate(labels[test_mask],
                                               clf.predict_proba(features[test_mask])[:, 1],
                                               multilabel=False))
            # clf = XGBClassifier(verbose=1)
            # clf.fit(features[train_data[mask_key]], labels[train_data[mask_key]],
            #         eval_set=[(features[val_data[mask_key]], labels[val_data[mask_key]])])

            print(classification_report(labels[train_mask],
                                        clf.predict(features[train_mask])))
            print(classification_report(labels[val_mask],
                                        clf.predict(features[val_mask])))
            print(classification_report(labels[test_mask],
                                        clf.predict(features[test_mask])))
        else:
            if len(labels.shape) == 1 or labels.shape[1] == 1:
                label_size = 1
                labels = labels.reshape(-1, 1)
                multilabel = False
            else:
                label_size = labels.shape[1]
                multilabel = True
            model = FCBaseline(in_nfeats=features.shape[1], labelsize=label_size, n_layers=1,
                               n_hidden=features.shape[1],
                               # n_hidden=128,
                               dropout=0.5)
            features = convert_matrix_to_numpy(features)
            features = torch.FloatTensor(features)
            if isinstance(labels, torch.Tensor):
                labels = labels.float()
            labels = torch.FloatTensor(labels)
            train_dataloader = FCDataloader().build_dataloader_non_graph(features[train_mask],
                                                                         labels[train_mask],
                                                                         batch_size=10000,
                                                                         # batch_size=None,
                                                                         shuffle=True,
                                                                         train_val_test="train")
            val_dataloader = FCDataloader().build_dataloader_non_graph(features[val_mask],
                                                                       labels[val_mask],
                                                                       batch_size=10000,
                                                                       # batch_size=None,
                                                                       shuffle=True,
                                                                       train_val_test="val")
            test_dataloader = FCDataloader().build_dataloader_non_graph(features[test_mask],
                                                                        labels[test_mask],
                                                                        batch_size=10000,
                                                                        # batch_size=None,
                                                                        shuffle=True,
                                                                        train_val_test="test")
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./models", "MLPVanilla", str(datetime.datetime.now())),
                # dirpath=os.path.join("./models", "MLPVanilla_nobatchnorm_fullbatch", str(datetime.datetime.now())),
                filename='epoch={epoch}-val_loss_epoch={'
                         'val/loss_epoch:.6f}-val_accuracy_epoch={'
                         'val/accuracy_epoch:.4f}-val_macro_f1_epoch={'
                         'val/macro_f1_epoch:.4f}-val_topk_precision_epoch={'
                         'val/topk_precision_epoch:.4f}-val_f1_anomaly_epoch={'
                         'val/f1_anomaly_epoch:.4f}',
                # monitor='val/best_macro_f1_epoch', mode="max", save_last=True,
                monitor='val/auc_epoch', mode="max", save_last=True,
                # monitor='val/macro_f1_epoch', mode="max", save_last=True,
                # monitor='val/ap_epoch', mode="max", save_last=True,
                # monitor='val/loss_epoch', mode="min", save_last=True,
                auto_insert_metric_name=False)
            earlystopping_callback = EarlyStopping(monitor='val/loss_epoch',
                                                   min_delta=0.00,
                                                   # patience=1,
                                                   patience=100,
                                                   verbose=True,
                                                   mode='min')
            if gpus is not None:
                accelerator = "gpu"
            else:
                accelerator = "cpu"
            clf = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=gpus,
                             callbacks=[checkpoint_callback, earlystopping_callback],
                             check_val_every_n_epoch=10, profiler=False,
                             move_metrics_to_cpu=True, precision=32)
            start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
            clf.fit(model, train_dataloader, [val_dataloader])
            end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
            print(f"Training time elapsed: {end_training_time - start_training_time} sec")
            print(checkpoint_callback.best_model_path)
            train_labels, train_prediction = zip(*clf.predict(dataloaders=train_dataloader,
                                                              ckpt_path=checkpoint_callback.best_model_path))
            val_labels, val_prediction = zip(*clf.predict(dataloaders=val_dataloader,
                                                          ckpt_path=checkpoint_callback.best_model_path))
            start_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
            test_labels, test_prediction = zip(*clf.predict(dataloaders=test_dataloader,
                                                            ckpt_path=checkpoint_callback.best_model_path))
            end_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
            print(f"Testing time elapsed: {end_testing_time - start_testing_time} sec")

            # if len(labels.shape) > 1 and labels.shape[1] > 1:
            train_labels = torch.cat(train_labels)
            val_labels = torch.cat(val_labels)
            test_labels = torch.cat(test_labels)
            train_prediction = torch.cat(train_prediction)
            val_prediction = torch.cat(val_prediction)
            test_prediction = torch.cat(test_prediction)
            print("train anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                  evaluate(train_labels,
                           train_prediction,
                           multilabel=multilabel))
            print("val anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                  evaluate(val_labels,
                           val_prediction,
                           multilabel=multilabel))
            print("test anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                  test_results := evaluate(test_labels,
                                           test_prediction,
                                           multilabel=multilabel))
            print(classification_report(train_labels,
                                        train_prediction > 0.5))
            print(classification_report(val_labels,
                                        val_prediction > 0.5))
            print(classification_report(test_labels,
                                        test_prediction > 0.5))

            # for w in train_dataloader._get_iterator()._workers:
            #     w.terminate()
            # train_dataloader._get_iterator()._shutdown = True
            # for w in val_dataloader._get_iterator()._workers:
            #     w.terminate()
            # val_dataloader._get_iterator()._shutdown = True
            # for w in test_dataloader._get_iterator()._workers:
            #     w.terminate()
            # test_dataloader._get_iterator()._shutdown = True


        return test_results