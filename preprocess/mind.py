import collections
import json
import random
import sys
from itertools import combinations

import pandas as pd
import os

import numpy as np

import CommonModules as CM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, \
    f1_score
from xgboost import XGBClassifier
import scipy.sparse

from global_parameters import MINDCountVectorizerDataFolder, MINDBipartiteCountVectorizerDataFolder
from models.GNNVanilla import evaluate

from preprocess._citation import transform_data, export_data, get_sampled_graph, get_datastats


def prepare_testing_sample(inputf, output_folder, reduced_dim=None, only_get_datastats=False, is_bipartite=False):
    feature_type = "count_vectorizer"
    min_frequency_label = 1000

    senders = []  # user
    receivers = []  # user
    reviews = []  # news feature
    # Helpful ratio <= ratio_threshold will be marked as bad review (1).
    # (0) if there is no vote or below vote_count_threshold. (-1) if ratio > ratio_threshold
    labels = []
    # train_news = {}
    # val_news = {}
    # test_news = {}
    # train_index = []
    # val_index = []
    # test_index = []
    news_user_dict = collections.defaultdict(set)
    news_feature_dict = {}
    news_label_dict = {}
    deduplicate_labels = []
    estimate_edge_size = 0
    # for dataset_name in ["_train", "_val", "_test"]:  # No longer use it since there are overlaps for news
    for file in CM.IO.ListFiles(inputf, ".tsv"):
        # if dataset_name in file:
        if "behaviors_" in file:
            with open(file) as fd:
                lines = fd.readlines()
            # if dataset_name == "_train":
            #     train_index = np.arange(len(lines), dtype=int)
            # if dataset_name == "_val":
            #     val_index = np.arange(len(lines), dtype=int)
            # if dataset_name == "_test":
            #     test_index = np.arange(len(lines), dtype=int)
            for line in lines:
                data = line.split("\t")
                user = data[1]
                newses = data[3].split()
                for news in newses:
                    news_user_dict[news].add(user)
        if "news_" in file:
            with open(file) as fd:
                lines = fd.readlines()
            for line in lines:
                data = line.split("\t")
                news = data[0]
                feature = data[3]
                label = data[1]
                news_feature_dict[news] = feature
                news_label_dict[news] = label
                # if dataset_name == "_train":
                #     train_news[news] = 1
                # if dataset_name == "_val" and news not in train_news:
                #     val_news[news] = 1
                # if dataset_name == "_test" and news not in train_news and news not in val_news:
                #     test_news[news] = 1
    for news in news_user_dict:
        if len(news_user_dict[news]) > 10:  # Sample up to 10 users
            news_user_dict[news] = random.sample(news_user_dict[news], 10)
        estimate_edge_size += len(news_user_dict[news]) * (len(news_user_dict[news]) - 1) // 2
    print("estimate_edge_size:", estimate_edge_size)
    for news in news_user_dict:
        user_pairs = combinations(news_user_dict[news], 2)
        deduplicate_labels.append(news_label_dict[news])
        for user_pair in user_pairs:  # Will only consider author pairs
            senders.append(user_pair[0])
            receivers.append(user_pair[1])
            reviews.append(news_feature_dict[news])
            labels.append(news_label_dict[news])
            # if news in train_news:
            #     train_index.append(len(senders) - 1)
            # if news in val_news:
            #     val_index.append(len(senders) - 1)
            # if news in test_news:
            #     test_index.append(len(senders) - 1)
    # train_index = np.array(train_index)
    # val_index = np.array(val_index)
    # test_index = np.array(test_index)
    # assert train_index.shape[0] + val_index.shape[0] + test_index.shape[0] == len(labels)
    if only_get_datastats:
        _reduced_dim = False
    else:
        _reduced_dim = reduced_dim
    transformed_data = transform_data(senders, receivers, reviews, labels,
                                      deduplicate_labels, min_frequency_label,
                                      feature_type, multilabel=True,
                                      randomize_by_node=True, reduced_dim=_reduced_dim)
    print("Dataset: MIND")
    get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, reduced_dim)

    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    # output_folder = os.path.join(MINDCountVectorizerDataFolder, "reduce_dim_256")
    # os.makedirs(output_folder, exist_ok=True)
    # prepare_testing_sample(os.path.join(MINDCountVectorizerDataFolder, ".."), output_folder,
    #                        reduced_dim=256, only_get_datastats=True)
    #
    # df_train = pd.read_csv(os.path.join(output_folder, "train_edges.csv"))
    # df_val = pd.read_csv(os.path.join(output_folder, "val_edges.csv"))
    # df_test = pd.read_csv(os.path.join(output_folder, "test_edges.csv"))
    # train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "train_embeds_labels.pkl"))
    # val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "val_embeds_labels.pkl"))
    # test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "test_embeds_labels.pkl"))
    # label_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "label_vectorizer.pkl"))
    # feature_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "feature_vectorizer.pkl"))
    # CM.Utilities.TimeElapsed(LastTime=True)
    # train_labels[train_labels < 0] = 0
    # val_labels[val_labels < 0] = 0
    # test_labels[test_labels < 0] = 0
    # CM.Utilities.TimeElapsed(LastTime=True)
    #
    # #################### Get sampled graph #######################
    # sample_sizes = [5000, 10000, 20000, 40000, 80000]
    # # Will auto add sth. like _bfs_sampled_size in the folder path
    # os.makedirs(output_folder, exist_ok=True)
    # get_sampled_graph(
    #     df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels,
    #     test_labels, label_vectorizer, feature_vectorizer, output_folder, sample_sizes=sample_sizes)
    # ##############################################################

    ##############################Bipartite graphs (Not work)###################################
    output_folder = os.path.join(MINDBipartiteCountVectorizerDataFolder, "reduce_dim_256")
    os.makedirs(output_folder, exist_ok=True)
    prepare_testing_sample(os.path.join(MINDBipartiteCountVectorizerDataFolder, ".."), output_folder,
                           reduced_dim=256, only_get_datastats=False, is_bipartite=True)

    ############################Simple baselines#################################
    # # clf = LogisticRegression(verbose=1, penalty="l1", solver="liblinear")
    # # clf = LogisticRegression(verbose=1, class_weight="balanced", max_iter=1000)
    # clf = LogisticRegression(verbose=1, max_iter=1000)
    # clf.fit(train_embeds, train_labels)
    #
    # # clf = XGBClassifier(verbose=1)
    # # clf.fit(train_embeds, train_labels, eval_set=[(val_embeds, val_labels)])
    #
    # train_pred_labels, train_pred_probs = clf.predict(train_embeds), clf.predict_proba(train_embeds)[:, 1]
    # val_pred_labels, val_pred_probs = clf.predict(val_embeds), clf.predict_proba(val_embeds)[:, 1]
    # test_pred_labels, test_pred_probs = clf.predict(test_embeds), clf.predict_proba(test_embeds)[:, 1]
    # print(classification_report(train_labels, train_pred_labels))
    # print(classification_report(val_labels, val_pred_labels))
    # print(classification_report(test_labels, test_pred_labels))
    # print("train anomaly F1, roc_auc_score, AP, macro_f1, micro_f1", evaluate(train_labels, train_pred_probs))
    # print("val anomaly F1, roc_auc_score, AP, macro_f1, micro_f1", evaluate(val_labels, val_pred_probs))
    # print("test anomaly F1, roc_auc_score, AP, macro_f1, micro_f1", evaluate(test_labels, test_pred_probs))
    # print("train finished", CM.Utilities.TimeElapsed(LastTime=True))