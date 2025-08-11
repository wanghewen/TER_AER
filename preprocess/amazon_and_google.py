import json
import pandas as pd
import os

import numpy as np

import CommonModules as CM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, \
    f1_score

from global_parameters import AmazonBipartiteTextDataFolder, GoogleLocalBipartiteTextDataFolder
from models.GNNVanilla import evaluate

from preprocess._citation import transform_data, get_sampled_graph, get_datastats, export_data


def prepare_testing_sample(inputf, output_folder, feature_type, review_type,
                           randomize_by_node=True, reduced_dim=None, only_get_datastats=False):
    assert feature_type in ["count_vectorizer", "sentence_transformer", "stemmed_count_vectorizer", "text"]
    assert review_type in ["Amazon", "Google"]

    senders = []  # Reviewer
    receivers = []  # Product
    reviews = []
    # Helpful ratio <= ratio_threshold will be marked as bad review (1).
    # (0) if there is no vote or below vote_count_threshold. (-1) if ratio > ratio_threshold
    labels = []
    deduplicate_labels = []
    with open(inputf) as fd:
        for line in fd:
            data = json.loads(line)
            if review_type == "Amazon":
                senders.append(data["reviewerID"])
                receivers.append(data["asin"])
                reviews.append(str(data["reviewText"]))
                rating = data["overall"]
            elif review_type == "Google":
                senders.append(data["user_id"])
                receivers.append(data["gmap_id"])
                reviews.append(str(data["text"]))
                rating = data["rating"]
            if rating <= 2:
                label = [-1]
            elif rating >= 4:
                label = [1]
            elif rating == 3:
                label = [0]
            else:
                raise ValueError
            labels.append(label)

    transformed_data = transform_data(
        senders, receivers, reviews, labels, None, None, feature_type, multilabel=True,
        randomize_by_node=randomize_by_node, reduced_dim=None)
    print(f"Dataset: {os.path.basename(inputf)}")
    get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, reduced_dim)

    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    # input_folder = AmazonBipartiteTextDataFolder
    # review_type = "Amazon"
    # input_folder = GoogleLocalBipartiteTextDataFolder
    # review_type = "Google"
    for input_folder, review_type in [(AmazonBipartiteTextDataFolder, "Amazon"),
                                      (GoogleLocalBipartiteTextDataFolder, "Google")]:
        os.makedirs(input_folder, exist_ok=True)
        for file in CM.IO.ListFiles(os.path.join(input_folder, ".."), ".json"):
            product_category = os.path.splitext(os.path.basename(file))[0]
            output_folder = os.path.join(input_folder, product_category, "base")
            os.makedirs(output_folder, exist_ok=True)
            prepare_testing_sample(os.path.join(input_folder, "..", file), output_folder,
                                   feature_type="text", review_type=review_type, randomize_by_node=False,
                                   reduced_dim=None, only_get_datastats=False)

            df_train = pd.read_csv(os.path.join(output_folder, "train_edges.csv"))
            df_val = pd.read_csv(os.path.join(output_folder, "val_edges.csv"))
            df_test = pd.read_csv(os.path.join(output_folder, "test_edges.csv"))
            train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "train_embeds_labels.pkl"))
            val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "val_embeds_labels.pkl"))
            test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(output_folder, "test_embeds_labels.pkl"))
            label_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "label_vectorizer.pkl"))
            feature_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "label_vectorizer.pkl"))

            #################### Get sampled graph #######################
            sample_sizes = [5000, 10000, 20000, 40000, 80000]
            # Will auto add sth. like _bfs_sampled_size in the folder path
            os.makedirs(output_folder, exist_ok=True)
            get_sampled_graph(
                df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels,
                test_labels, label_vectorizer, feature_vectorizer, output_folder, sample_sizes=sample_sizes)
            ##############################################################

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