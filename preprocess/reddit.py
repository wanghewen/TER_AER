import json
import pandas as pd
import os

import numpy as np

import CommonModules as CM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, \
    f1_score
from xgboost import XGBClassifier

from global_parameters import RedditCountVectorizerDataFolder
from models.GNNVanilla import evaluate

from preprocess._citation import transform_data, export_data, get_datastats


def prepare_testing_sample(inputf, output_folder, only_get_datastats=False):
    feature_type = "standard_scaler"

    senders = []  # Reviewer
    receivers = []  # Product
    reviews = []
    # Helpful ratio <= ratio_threshold will be marked as bad review (1).
    # (0) if there is no vote or below vote_count_threshold. (-1) if ratio > ratio_threshold
    labels = []
    for file in CM.IO.ListFiles(inputf, ".tsv"):
        df = pd.read_csv(file, sep="\t")
        for line in df.itertuples(index=False):
            data = line
            senders.append(data.SOURCE_SUBREDDIT)
            receivers.append(data.TARGET_SUBREDDIT)
            feature = [float(element) for element in data.PROPERTIES.split(",")]
            reviews.append(feature)
            label = data.LINK_SENTIMENT
            if label == -1:
                label = 1
            elif label == 1:
                label = 0
            else:
                raise ValueError
            labels.append(label)

    _reduced_dim = 99999
    transformed_data = transform_data(senders, receivers, reviews, labels, None, None,
                                      feature_type, multilabel=False, min_df=None,
                                      randomize_by_node=True)
    print("Dataset: Reddit")
    get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, _reduced_dim)
    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    counter = 0
    # output_folder = "/data/PublicGraph/AmazonReview/data_pos_ratio=0.3/data_count_vectorizer_minmax"
    output_folder = RedditCountVectorizerDataFolder
    os.makedirs(output_folder, exist_ok=True)
    prepare_testing_sample(os.path.join(RedditCountVectorizerDataFolder, ".."), output_folder, only_get_datastats=False)
    # output_folder = "/data/PublicGraph/AmazonReview/data_sentence_transformer"
    # os.makedirs(output_folder, exist_ok=True)
    # prepare_testing_sample("/data/PublicGraph/AmazonReview/reviews_Movies_and_TV_5.json", output_folder, feature_type="sentence_transformer")
    # output_folder = "/data/PublicGraph/AmazonReview/data_stemmed_count_vectorizer"
    # os.makedirs(output_folder, exist_ok=True)
    # prepare_testing_sample("/data/PublicGraph/AmazonReview/reviews_Movies_and_TV_5.json", output_folder, feature_type="stemmed_count_vectorizer")

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