import itertools
import json
from itertools import combinations

import pandas as pd
import os

import CommonModules as CM
from json import JSONDecodeError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from global_parameters import OAGCountVectorizerDataFolder, OAGBipartiteTextDataFolder
from models.GNNVanilla import evaluate
from preprocess._citation import transform_data, get_sampled_graph, get_datastats, export_data


def prepare_testing_sample(input_folder, output_folder, feature_type, process_abstract=True, randomize_by_node=True,
                           reduced_dim=None, only_get_datastats=False, is_bipartite=False):
    """

    :param input_folder:
    :param output_folder:
    :param feature_type:
    :param process_abstract: Whehter to process abstract as attributed edges. If false, also save the memory for storing it.
    :return:
    """
    min_frequency_label = 1000  # Number of helpful votes needed for giving labels
    assert feature_type in ["count_vectorizer", "sentence_transformer", "stemmed_count_vectorizer", "text"]

    senders = []  # Reviewer
    receivers = []  # Product
    reviews = []
    # Helpful ratio <= ratio_threshold will be marked as bad review (1).
    # (0) if there is no vote or below vote_count_threshold. (-1) if ratio > ratio_threshold
    labels = []
    deduplicate_labels = []
    print("Generating nodes and edges...")
    CM.Utilities.TimeElapsed(LastTime=True)
    # for inputf in CM.IO.ListFiles(input_folder, ".txt"):
    for inputf in CM.IO.ListFiles(input_folder, ".txt")[:1]:
        with open(inputf) as fd:
            for line in fd:
                data = json.loads(line)
                if "indexed_abstract" not in data or "authors" not in data or "fos" not in data:
                    continue
                if is_bipartite:
                    if "venue" not in data or "id" not in data["venue"]:
                        continue
                label = [element["name"] for element in data["fos"]]
                deduplicate_labels.append(label)
                if process_abstract:
                    try:
                        abstract = json.loads(data["indexed_abstract"])["InvertedIndex"]
                    except JSONDecodeError:  # Some unknown characters in the abstract
                        continue
                    max_abstract_index = max(itertools.chain(*abstract.values()))
                    new_abstract = [0] * (max_abstract_index + 1)
                    for word, indexes in abstract.items():
                        for index in indexes:
                            new_abstract[index] = word
                    abstract = new_abstract
                    new_abstract = [word for word in abstract if word != 0]
                    abstract = " ".join(new_abstract)
                else:
                    abstract = ""
                if is_bipartite:
                    authors = [element["id"] for element in data["authors"] if "id" in element]
                    venue = data["venue"]["id"]
                    for author in authors:  # Will only consider author pairs
                        senders.append(author)
                        receivers.append(venue)
                        reviews.append(abstract)
                        labels.append(label)
                else:
                    authors = [element["id"] for element in data["authors"] if "id" in element]
                    if len(authors) > 10:  # Drop paper with too many authors
                        continue
                    author_pairs = combinations(authors, 2)
                    for author_pair in author_pairs:  # Will only consider author pairs
                        senders.append(author_pair[0])
                        receivers.append(author_pair[1])
                        reviews.append(abstract)
                        labels.append(label)
    print("Read files finished in:", CM.Utilities.TimeElapsed(LastTime=True))
    # print("Removing classes with too few labels...")
    # label_vectorizer = MultiLabelBinarizer(sparse_output=True)
    # deduplicate_labels = label_vectorizer.fit_transform(deduplicate_labels)
    # CM.IO.ExportToPkl(os.path.join(output_folder, "label_vectorizer_deduplicate_reviews.pkl"), label_vectorizer)
    if only_get_datastats:
        _reduced_dim = False
    else:
        _reduced_dim = reduced_dim
    transformed_data = transform_data(
        senders, receivers, reviews, labels, deduplicate_labels, min_frequency_label, feature_type,
        randomize_by_node=randomize_by_node, reduced_dim=_reduced_dim)
    print("Dataset: OAG21")
    get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, reduced_dim)

    export_data(*transformed_data, output_folder=output_folder)
    return


if __name__ == '__main__':
    # # output_folder = OAGCountVectorizerDataFolder
    # # os.makedirs(output_folder, exist_ok=True)
    # # prepare_testing_sample(os.path.join(OAGCountVectorizerDataFolder, ".."), output_folder,
    # #                        feature_type="count_vectorizer")
    # output_folder = os.path.join(OAGCountVectorizerDataFolder, "reduce_dim_256")
    # os.makedirs(output_folder, exist_ok=True)
    # prepare_testing_sample(os.path.join(OAGCountVectorizerDataFolder, ".."), output_folder,
    #                        feature_type="count_vectorizer", reduced_dim=256, only_get_datastats=False)

    ##############################Bipartite graphs###################################
    output_folder = os.path.join(OAGBipartiteTextDataFolder, "base")
    os.makedirs(output_folder, exist_ok=True)
    prepare_testing_sample(os.path.join(OAGBipartiteTextDataFolder, ".."), output_folder,
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
    clf.fit(train_embeds, train_labels.A)

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
