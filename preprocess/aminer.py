import json
from itertools import combinations

import pandas as pd
import os

import CommonModules as CM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from global_parameters import AMinerCountVectorizerDataFolder, AMinerBipartiteTextDataFolder
from models.GNNVanilla import evaluate
from preprocess._citation import transform_data, get_sampled_graph, get_datastats, export_data


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
    # for inputf in CM.IO.ListFiles(input_folder, ".txt"):
    for inputf in CM.IO.ListFiles(input_folder, ".txt")[:1]:
        with open(inputf) as fd:
            for line in fd:
                data = json.loads(line)
                if is_bipartite:
                    if "abstract" not in data or "keywords" not in data or "authors" not in data or "venue" not in data:
                        continue
                    if "id" not in data["venue"]:
                        continue
                    authors = [element["id"] for element in data["authors"] if "id" in element]
                    abstract = data["abstract"]
                    venue = data["venue"]["id"]
                    label = data["keywords"]
                    deduplicate_labels.append(label)
                    for author in authors:  # Will only consider author pairs
                        senders.append(author)
                        receivers.append(venue)
                        reviews.append(abstract)
                        labels.append(label)
                else:
                    if "abstract" not in data or "keywords" not in data:
                        continue
                    authors = [element["id"] for element in data["authors"] if "id" in element]
                    if len(authors) > 10:  # Drop paper with too many authors
                        continue
                    author_pairs = combinations(authors, 2)
                    label = data["keywords"]
                    deduplicate_labels.append(label)
                    for author_pair in author_pairs:  # Will only consider author pairs
                        senders.append(author_pair[0])
                        receivers.append(author_pair[1])
                        reviews.append(data["abstract"])
                        labels.append(label)
    CM.Utilities.TimeElapsed(LastTime=True)

    if only_get_datastats:
        _reduced_dim = False
    else:
        _reduced_dim = reduced_dim
    transformed_data = transform_data(
        senders, receivers, reviews, labels, deduplicate_labels, min_frequency_label, feature_type,
        randomize_by_node=randomize_by_node, reduced_dim=_reduced_dim)
    print("Dataset: AMiner")
    get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, reduced_dim)

    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    # output_folder = os.path.join(AMinerCountVectorizerDataFolder, "reduce_dim_256")
    # os.makedirs(output_folder, exist_ok=True)
    # prepare_testing_sample(os.path.join(AMinerCountVectorizerDataFolder, ".."), output_folder,
    #                        feature_type="count_vectorizer", randomize_by_node=True, reduced_dim=256,
    #                        only_get_datastats=False)

    ##############################Bipartite graphs###################################
    output_folder = os.path.join(AMinerBipartiteTextDataFolder, "base")
    os.makedirs(output_folder, exist_ok=True)
    prepare_testing_sample(os.path.join(AMinerBipartiteTextDataFolder, ".."), output_folder,
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
    feature_vectorizer = CM.IO.ImportFromPkl(os.path.join(output_folder, "feature_vectorizer.pkl"))
    print("label_vectorizer.classes_[:50]", label_vectorizer.classes_[:50])
    CM.Utilities.TimeElapsed(LastTime=True)
    #################### Get sampled graph #######################
    sample_sizes = [5000, 10000, 20000, 40000, 80000]
    # Will auto add sth. like _bfs_sampled_size in the folder path
    os.makedirs(output_folder, exist_ok=True)
    get_sampled_graph(
        df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels,
        test_labels, label_vectorizer, feature_vectorizer, output_folder, sample_sizes=sample_sizes)
    ##############################################################

    # clf = LogisticRegression(verbose=1, penalty="l1", solver="liblinear")
    # clf = LogisticRegression(verbose=1, class_weight="balanced", max_iter=1000)
    clf = LogisticRegression(verbose=1)
    clf.fit(train_embeds, train_labels)

    # clf = XGBClassifier(verbose=1)
    # clf.fit(train_embeds, train_labels, eval_set=[(val_embeds, val_labels)])

    train_pred_labels, train_pred_probs = clf.predict(train_embeds), clf.predict_proba(train_embeds)
    val_pred_labels, val_pred_probs = clf.predict(val_embeds), clf.predict_proba(val_embeds)
    test_pred_labels, test_pred_probs = clf.predict(test_embeds), clf.predict_proba(test_embeds)
    print(classification_report(train_labels, train_pred_labels))
    print(classification_report(val_labels, val_pred_labels))
    print(classification_report(test_labels, test_pred_labels))
    # TODO: Modify evaluate to support (multiclass_or_multilabel target)
    print(*evaluate(train_labels, train_pred_probs, subset="train"))
    print(*evaluate(val_labels, val_pred_probs, subset="val"))
    print(*evaluate(test_labels, test_pred_probs, subset="test"))
    print("train finished", CM.Utilities.TimeElapsed(LastTime=True))
