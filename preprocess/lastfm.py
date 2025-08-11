import collections
import json
from itertools import combinations
import dateutil.parser as dparser

import pandas as pd
import os

import numpy as np

import CommonModules as CM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from models.GNNVanilla import evaluate

from preprocess._citation import transform_data, export_data


def prepare_testing_sample(inputf, output_folder):
    feature_type = "standard_scaler"

    senders = []  # artist
    receivers = []  # artist
    reviews = []  # user feature
    labels = []
    train_index = []
    val_index = []
    test_index = []
    user_artist_dict = collections.defaultdict(set)
    user_feature_dict = {}
    user_label_dict = {}
    deduplicate_labels = []
    estimate_edge_size = 0

    with open(os.path.join(inputf, "usersha1-artmbid-artname-plays.tsv")) as fd:
        lines = fd.readlines()
    # if dataset_name == "_train":
    #     train_index = np.arange(len(lines), dtype=int)
    # if dataset_name == "_val":
    #     val_index = np.arange(len(lines), dtype=int)
    # if dataset_name == "_test":
    #     test_index = np.arange(len(lines), dtype=int)
    for line_index, line in enumerate(lines):
        data = line.split("\t")
        user = data[0]
        artist = data[1]
        if len(user) != 40 or len(artist) != 36:  # Sanity check
            continue
        user_artist_dict[user].add(artist)

    with open(os.path.join(inputf, "usersha1-profile.tsv")) as fd:
        lines = fd.readlines()
    nationality_features = []
    for line in lines:
        data = line.split("\t")
        nationality_features.append([data[3]])
    oneHotEncoder = OneHotEncoder(sparse=False)
    nationality_features = oneHotEncoder.fit_transform(nationality_features)
    nationality_features = nationality_features.tolist()
    for line, nationality_feature in zip(lines, nationality_features):
        data = line.split("\t")
        user = data[0]
        if data[2] == "":  # Impute missing values
            data[2] = 0
        age_feature = float(data[2])
        signup_feature = dparser.parse(data[4]).timestamp()
        label = data[1]
        user_feature_dict[user] = [age_feature] + nationality_feature + [signup_feature]
        if label == "m":
            label = -1
        elif label == "f":
            label = 1
        else:
            label = 0
        user_label_dict[user] = label

    train_user, test_user = train_test_split(list(user_label_dict.keys()), test_size=0.2, random_state=12)
    train_user, val_user = train_test_split(train_user, test_size=0.2, random_state=12)
    train_user = {user: 1 for user in train_user}
    val_user = {user: 1 for user in val_user}
    test_user = {user: 1 for user in test_user}

    for user in user_artist_dict:
        estimate_edge_size += len(user_artist_dict[user]) * (len(user_artist_dict[user]) - 1) // 2
    print("estimate_edge_size:", estimate_edge_size)
    for user in user_artist_dict:
        artist_pairs = combinations(user_artist_dict[user], 2)
        deduplicate_labels.append(user_label_dict[user])
        for artist_pair in artist_pairs:  # Will only consider author pairs
            senders.append(artist_pair[0])
            receivers.append(artist_pair[1])
            reviews.append(user_feature_dict[user])
            labels.append(user_label_dict[user])
            if label != 0:  # Exclude missing genders(labels)
                if user in train_user:
                    train_index.append(len(senders) - 1)
                if user in val_user:
                    val_index.append(len(senders) - 1)
                if user in test_user:
                    test_index.append(len(senders) - 1)
    train_index = np.array(train_index)
    val_index = np.array(val_index)
    test_index = np.array(test_index)
    assert train_index.shape[0] + val_index.shape[0] + test_index.shape[0] == len(labels)
    transformed_data = transform_data(senders, receivers, reviews, labels,
                                      deduplicate_labels, None,
                                      feature_type, multilabel=False, min_df=None,
                                      randomize_by_node=False, train_test_index=[train_index, val_index, test_index])
    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    counter = 0
    # output_folder = "/data/PublicGraph/AmazonReview/data_pos_ratio=0.3/data_count_vectorizer_minmax"
    output_folder = "/data/PublicGraph/LastFM/data_count_vectorizer"
    os.makedirs(output_folder, exist_ok=True)
    prepare_testing_sample("/data/PublicGraph/LastFM", output_folder)

    df_train = pd.read_csv(os.path.join(output_folder, "train_edges.csv"))
    df_val = pd.read_csv(os.path.join(output_folder, "val_edges.csv"))
    df_test = pd.read_csv(os.path.join(output_folder, "test_edges.csv"))
    train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "train_embeds_labels.pkl"))
    val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "val_embeds_labels.pkl"))
    test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "test_embeds_labels.pkl"))
    train_labels[train_labels < 0] = 0
    val_labels[val_labels < 0] = 0
    test_labels[test_labels < 0] = 0
    CM.Utilities.TimeElapsed(LastTime=True)

    # clf = LogisticRegression(verbose=1, penalty="l1", solver="liblinear")
    # clf = LogisticRegression(verbose=1, class_weight="balanced", max_iter=1000)
    clf = LogisticRegression(verbose=1, max_iter=1000)
    clf.fit(train_embeds, train_labels)

    # clf = XGBClassifier(verbose=1)
    # clf.fit(train_embeds, train_labels, eval_set=[(val_embeds, val_labels)])

    train_pred_labels, train_pred_probs = clf.predict(train_embeds), clf.predict_proba(train_embeds)[:, 1]
    val_pred_labels, val_pred_probs = clf.predict(val_embeds), clf.predict_proba(val_embeds)[:, 1]
    test_pred_labels, test_pred_probs = clf.predict(test_embeds), clf.predict_proba(test_embeds)[:, 1]
    print(classification_report(train_labels, train_pred_labels))
    print(classification_report(val_labels, val_pred_labels))
    print(classification_report(test_labels, test_pred_labels))
    print(*evaluate(train_labels, train_pred_probs, subset="train"))
    print(*evaluate(val_labels, val_pred_probs, subset="val"))
    print(*evaluate(test_labels, test_pred_probs, subset="test"))
    print("train finished", CM.Utilities.TimeElapsed(LastTime=True))