import json
import sys

import mmap
from itertools import combinations

import pandas as pd
import os
import numpy as np

import CommonModules as CM
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

from global_parameters import DBLPCountVectorizerDataFolder, DBLPBipartiteTextDataFolder
from models.GNNVanilla import evaluate
from preprocess._citation import transform_data, export_data, get_sampled_graph, get_datastats


def prepare_testing_sample(input_folder, output_folder, feature_type, randomize_by_node=True, reduced_dim=None,
                           only_get_datastats=False, is_bipartite=False):
    min_frequency_label = 1000  # Number of labels across documents that will be selected as condidate labels
    assert feature_type in ["count_vectorizer", "sentence_transformer", "stemmed_count_vectorizer", "text"]

    senders = []  # Reviewer
    receivers = []  # Product
    reviews = []
    # Helpful ratio <= ratio_threshold will be marked as bad review (1).
    # (0) if there is no vote or below vote_count_threshold. (-1) if ratio > ratio_threshold
    labels = []
    deduplicate_labels = []
    CM.Utilities.TimeElapsed(LastTime=True)
    with open(os.path.join(input_folder, "dblpv13.json")) as fd:
        all_data = []
        for line in fd:  # Slow but more memory efficient
            all_data.append(re.sub(r"NumberInt\((\d+)\)", r"\1", line))
        all_data = "".join(all_data)
        all_data = json.loads(all_data)
    print("read data finished in:", CM.Utilities.TimeElapsed(LastTime=True))
    for data in all_data:
        if is_bipartite:
            if "abstract" not in data or "fos" not in data or "authors" not in data or "venue" not in data:
                continue
            if "_id" not in data["venue"]:
                continue
            authors = [element["_id"] for element in data["authors"] if "_id" in element]
            abstract = data["abstract"]
            venue = data["venue"]["_id"]
            label = data["fos"]
            deduplicate_labels.append(label)
            for author in authors:  # Will only consider author pairs
                senders.append(author)
                receivers.append(venue)
                reviews.append(abstract)
                labels.append(label)
        else:
            if "abstract" not in data or "fos" not in data or "authors" not in data:
                continue
            authors = [element["_id"] for element in data["authors"] if "_id" in element]
            if len(authors) > 10:  # Drop paper with too many authors
                continue
            author_pairs = combinations(authors, 2)
            label = data["fos"]
            deduplicate_labels.append(label)
            for author_pair in author_pairs:  # Will only consider author pairs
                senders.append(author_pair[0])
                receivers.append(author_pair[1])
                reviews.append(data["abstract"])
                labels.append(label)
    print("parse data finished in:", CM.Utilities.TimeElapsed(LastTime=True))
    if only_get_datastats:
        _reduced_dim = False
    else:
        _reduced_dim = reduced_dim
    transformed_data = transform_data(
        senders, receivers, reviews, labels, deduplicate_labels, min_frequency_label, feature_type,
        multilabel=True, randomize_by_node=randomize_by_node, reduced_dim=_reduced_dim)
    print("Dataset: DBLP")
    get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, reduced_dim)

    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    # # output_folder = "/data/PublicGraph/DBLP/data_count_vectorizer"
    # # output_folder = DBLPCountVectorizerDataFolder
    # # os.makedirs(output_folder, exist_ok=True)
    # # prepare_testing_sample(os.path.join(DBLPCountVectorizerDataFolder, ".."), output_folder,
    # #                        feature_type="count_vectorizer", randomize_by_node=True)
    # output_folder = os.path.join(DBLPCountVectorizerDataFolder, "reduce_dim_256")
    # os.makedirs(output_folder, exist_ok=True)
    # prepare_testing_sample(os.path.join(DBLPCountVectorizerDataFolder, ".."), output_folder,
    #                        feature_type="count_vectorizer", randomize_by_node=True, reduced_dim=256,
    #                        only_get_datastats=True)

    ##############################Bipartite graphs###################################
    output_folder = os.path.join(DBLPBipartiteTextDataFolder, "base")
    os.makedirs(output_folder, exist_ok=True)
    prepare_testing_sample(os.path.join(DBLPBipartiteTextDataFolder, ".."), output_folder,
                           feature_type="text", randomize_by_node=False,
                           reduced_dim=None, only_get_datastats=False, is_bipartite=True)
    ############################################################################################

    df_train = pd.read_csv(os.path.join(output_folder, "train_edges.csv"))
    df_val = pd.read_csv(os.path.join(output_folder, "val_edges.csv"))
    df_test = pd.read_csv(os.path.join(output_folder, "test_edges.csv"))
    train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "train_embeds_labels.pkl"))
    val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "val_embeds_labels.pkl"))
    test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "test_embeds_labels.pkl"))
    label_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "label_vectorizer.pkl"))
    feature_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "label_vectorizer.pkl"))
    print("label_vectorizer.classes_[:50]", label_vectorizer.classes_[:50])
    #################### Get sampled graph #######################
    sample_sizes = [5000, 10000, 20000, 40000, 80000]
    # Will auto add sth. like _bfs_sampled_size in the folder path
    os.makedirs(output_folder, exist_ok=True)
    get_sampled_graph(
        df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels,
        test_labels, label_vectorizer, feature_vectorizer, output_folder, sample_sizes=sample_sizes)
    ##############################################################

    #################Not work for text features #############
    # train_size = train_embeds.shape[0] // 10
    train_size = train_embeds.shape[0]
    val_size = val_embeds.shape[0]
    test_size = test_embeds.shape[0]
    train_embeds, val_embeds, test_embeds = train_embeds[:train_size], val_embeds[:val_size], test_embeds[:test_size]
    # train_embeds, val_embeds, test_embeds = train_embeds[:train_size, :2], val_embeds[:val_size, :2], test_embeds[:test_size, :2]  # For debug purpose, reduce feature dimension
    train_labels, val_labels, test_labels = train_labels[:train_size], val_labels[:val_size], test_labels[:test_size]
    train_labels, val_labels, test_labels = train_labels[:, :50].A, val_labels[:, :50].A, test_labels[:, :50].A
    CM.Utilities.TimeElapsed(LastTime=True)

    # clf = LogisticRegression(verbose=1, penalty="l1", solver="liblinear")
    # clf = LogisticRegression(verbose=1, class_weight="balanced", max_iter=1000)
    # clf = MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000, verbose=1), n_jobs=10)
    clf = MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000, verbose=1))
    # clf = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1, multi_class="ovr")
    clf.fit(train_embeds, train_labels)

    # clf = XGBClassifier(verbose=1)
    # clf.fit(train_embeds, train_labels, eval_set=[(val_embeds, val_labels)])

    train_pred_labels, train_pred_probs = clf.predict(train_embeds), clf.predict_proba(train_embeds)
    val_pred_labels, val_pred_probs = clf.predict(val_embeds), clf.predict_proba(val_embeds)
    test_pred_labels, test_pred_probs = clf.predict(test_embeds), clf.predict_proba(test_embeds)
    train_pred_probs = np.array(train_pred_probs)[:, :, 1].transpose()
    val_pred_probs = np.array(val_pred_probs)[:, :, 1].transpose()
    test_pred_probs = np.array(test_pred_probs)[:, :, 1].transpose()
    print(classification_report(train_labels, train_pred_labels))
    print(classification_report(val_labels, val_pred_labels))
    print(classification_report(test_labels, test_pred_labels))
    # TODO: Modify evaluate to support (multiclass_or_multilabel target)
    print("train anomaly F1, roc_auc_score, AP, macro_f1, micro_f1", evaluate(train_labels, train_pred_probs, multiclass_or_multilabel=True))
    print("val anomaly F1, roc_auc_score, AP, macro_f1, micro_f1", evaluate(val_labels, val_pred_probs, multiclass_or_multilabel=True))
    print("test anomaly F1, roc_auc_score, AP, macro_f1, micro_f1", evaluate(test_labels, test_pred_probs, multiclass_or_multilabel=True))
    print("train finished", CM.Utilities.TimeElapsed(LastTime=True))