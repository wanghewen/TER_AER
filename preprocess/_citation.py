import sys

import nltk.stem
import os
import scipy.sparse
import torch
from scipy.linalg import clarkson_woodruff_transform
from scipy.sparse.csgraph import breadth_first_order
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import CommonModules as CM
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from transformers import BertModel
from transformers import AutoTokenizer


english_stemmer = nltk.stem.SnowballStemmer('english')



class StemmedCountVectorizer(CountVectorizer):
    """
    Class used to build the stemmed count vectorizer english analyzer
    -----
    Methods
    -----
    build_analyzer(self)
        builds the english analyzer
    """
    def build_analyzer(self):
        """
        builds the english analyzer
        Returns
        -------
        lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
            the english analyzer for the doc
        """
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def get_datastats(senders, receivers, reviews, transformed_data, only_get_datastats, reduced_dim):
    df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels, test_labels, \
    label_vectorizer, feature_vectorizer = transformed_data
    print("Number of nodes:", len(set(senders) | set(receivers)))
    print("Number of edges:", len(reviews))
    if only_get_datastats:
        if len(train_embeds.shape) > 1:
            print("Original feature dimension:", train_embeds.shape[1])
    if len(train_embeds.shape) > 1:
        if reduced_dim is None:
            reduced_dim = train_embeds.shape[1]
        print("New feature dimension:", min(reduced_dim, train_embeds.shape[1]))
    if scipy.sparse.issparse(train_embeds):
        nnz = train_embeds.count_nonzero() + val_embeds.count_nonzero() + test_embeds.count_nonzero()
    else:
        nnz = np.count_nonzero(train_embeds) + np.count_nonzero(val_embeds) + np.count_nonzero(test_embeds)
    print("New feature nnz:", nnz)
    # print("Original label dimension:", len(set(sum(labels, []))))  # Too slow
    if len(train_labels.shape) > 1:
        print("New label dimension:", train_labels.shape[1])
    else:
        print("New label dimension:", 1)
    if only_get_datastats:
        sys.exit()


def transform_data(senders, receivers, reviews, labels, deduplicate_labels, min_frequency_label,
                   feature_type, multilabel=True, min_df=50, randomize_by_node=True,
                   train_test_index=None, reduced_dim=None):
    """
    Transform features and labels into vector form. Then generate train/val/test datasets

    :param senders: Sender node list
    :param receivers: Receiver node list
    :param reviews: Edge feature list
    :param labels: Label list, for non-multiclass_or_multilabel data
    :param deduplicate_labels: The original count of the number of labels in the dataset. For multiclass_or_multilabel data,
    provide this with min_frequency_label to filter out too little labels
    :param min_frequency_label: For multiclass_or_multilabel data, provide to filter out too little labels
    :param feature_type: The way to convert edge features
    :param multilabel: Indicate if the data is multiclass_or_multilabel dataset
    :param min_df: min_df for CountVectorizer
    :param randomize_by_node: Whether to split the dataset into train/test/val by node or edge
    :param train_test_index: Provide train/test index in advance to replace randomly splitting
    :param reduced_dim: Feature vector dimension for clarkson_woodruff_transform
    :return:
    """
    # label_vectorizer = OneHotEncoder(min_frequency=5)
    if multilabel:
        if deduplicate_labels is None:
            deduplicate_labels = labels
        if min_frequency_label is None:
            min_frequency_label = 0
        label_vectorizer = MultiLabelBinarizer(sparse_output=True)
        deduplicate_labels = label_vectorizer.fit_transform(deduplicate_labels)
        classes = []
        class_mask = (deduplicate_labels.sum(0).A >= min_frequency_label).reshape(-1).tolist()
        class_counts = deduplicate_labels.sum(0).A.reshape(-1)[class_mask].tolist()
        for index in range(len(label_vectorizer.classes_)):
            if class_mask[index]:
                classes.append(label_vectorizer.classes_[index])
        class_counts, classes = zip(*sorted(zip(class_counts, classes), reverse=True))
        label_vectorizer = MultiLabelBinarizer(sparse_output=True, classes=classes)
        print("classes[:1000], class_counts[:1000]", classes[:1000], class_counts[:1000])
        labels = label_vectorizer.fit_transform(labels)
        # labels = labels[:, (labels.sum(0).A >= min_frequency_label).reshape(-1)]
        print("labels.shape:", labels.shape)
        print("# of senders:", len(set(senders)), "# of receivers:", len(set(receivers)),
              "# of different abstract:", len(set(reviews)), "# of edges/abstract:", len(reviews))
        print("label_vectorizer.fit_transform:", CM.Utilities.TimeElapsed(LastTime=True))
    else:
        label_vectorizer = None
        labels = np.array(labels, dtype=int)
        print("labels count:", dict(zip(*np.asarray(np.unique(labels, return_counts=True)).tolist())))

    # Should not use vectorizer like TF-IDF here since reviews are duplicated in the above steps
    if feature_type == "count_vectorizer":
        feature_vectorizer = CountVectorizer(min_df=min_df)
        features = feature_vectorizer.fit_transform(reviews)
        # features: scipy.sparse.csr_matrix
        # column_max = features.max(0).A.reshape(-1)
        # features = features @ scipy.sparse.diags(column_max)
    elif feature_type == "stemmed_count_vectorizer":
        feature_vectorizer = StemmedCountVectorizer(min_df=min_df)
        features = feature_vectorizer.fit_transform(reviews)
    elif feature_type == "sentence_transformer":
        feature_vectorizer = SentenceTransformer('all-mpnet-base-v2')
        features = feature_vectorizer.encode(reviews, show_progress_bar=True, batch_size=128)
    elif feature_type in ["bert-base-uncased", "prajjwal1/bert-tiny"]:
        model_name_or_path = feature_type
        feature_vectorizer = BertModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        features_list = []
        for reviews in batch(reviews, 4096):
            encoded_text = tokenizer.batch_encode_plus(reviews, max_length=64, padding='max_length', truncation=True)
            token_edges = torch.LongTensor(encoded_text["input_ids"])
            attention_edges = torch.LongTensor(encoded_text["attention_mask"])

            hidden_states = feature_vectorizer(token_edges, attention_edges)
            last_hidden_states = hidden_states[0]
            embedding_output = last_hidden_states[:, 0, :]  # use cls embedding here, but may use other pooling operation

            # features_list.append(feature_vectorizer(input_ids=token_edges).mean(1).detach().cpu().numpy())
            features_list.append(embedding_output.detach().cpu().numpy())
        features = np.concatenate(features_list)
    elif feature_type == "standard_scaler":
        feature_vectorizer = StandardScaler()
        features = feature_vectorizer.fit_transform(reviews)
    elif feature_type == "text":
        feature_vectorizer = None
        features = np.array(reviews, dtype=object)
    else:
        raise TypeError
    if reduced_dim:
        print("Start doing Clarkson_woodruff_transform... reduced dim:", reduced_dim)
        features_T = clarkson_woodruff_transform(features.T, sketch_size=reduced_dim)
        features = features_T.T

    index = np.arange(labels.shape[0])
    if train_test_index is None:
        train_index, test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
        train_index, val_index = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=12)
    else:
        train_index, val_index, test_index = train_test_index
    df = pd.DataFrame({'SENDER_id': senders, 'RECEIVER_id': receivers})
    print("feature_vectorizer.fit_transform", CM.Utilities.TimeElapsed(LastTime=True))
    print("features.shape:", features.shape)

    nodeid_dict = {}
    for sndr_id, rcvr_id in zip(df["SENDER_id"], df["RECEIVER_id"]):
        if sndr_id not in nodeid_dict:
            nodeid_dict[sndr_id] = len(nodeid_dict)
        if rcvr_id not in nodeid_dict:
            nodeid_dict[rcvr_id] = len(nodeid_dict)
    df["SENDER_id"] = df["SENDER_id"].apply(lambda account, account_nodeid_dict: account_nodeid_dict[account], account_nodeid_dict=nodeid_dict)
    df["RECEIVER_id"] = df["RECEIVER_id"].apply(lambda account, account_nodeid_dict: account_nodeid_dict[account], account_nodeid_dict=nodeid_dict)


    print("encode sender and receivers", CM.Utilities.TimeElapsed(LastTime=True))
    df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
    train_embeds, val_embeds, test_embeds = features[train_index], features[val_index], features[test_index]
    train_labels, val_labels, test_labels = labels[train_index], labels[val_index], labels[test_index]

    if randomize_by_node:
        print("Randomly split dataset by nodes...")
        df = pd.concat([df_train, df_val, df_test])
        if isinstance(train_embeds, np.ndarray):
            embeds = np.concatenate([train_embeds, val_embeds, test_embeds], axis=0)
        else:
            embeds = scipy.sparse.vstack([train_embeds, val_embeds, test_embeds])
        if isinstance(train_labels, np.ndarray):
            labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)
        else:
            labels = scipy.sparse.vstack([train_labels, val_labels, test_labels])
        all_node_list = np.array(list(set(df["SENDER_id"]) | set(df["RECEIVER_id"])))
        index = np.arange(len(all_node_list))
        node_train_index, node_test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
        node_train_index, node_val_index = train_test_split(node_train_index, test_size=0.2, shuffle=True,
                                                            random_state=12)
        train_index = (df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
            all_node_list[node_train_index])).to_numpy()
        val_index = ((df["SENDER_id"].isin(all_node_list[node_val_index]) | df["RECEIVER_id"].isin(
            all_node_list[node_val_index])) & ~(
                df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
            all_node_list[node_train_index]))).to_numpy()
        # test_index = ((df["SENDER_id"].isin(all_node_list[node_test_index]) | df["RECEIVER_id"].isin(
        #     all_node_list[node_test_index])) & ~(
        #         df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
        #     all_node_list[node_train_index]))).to_numpy()  # Not remove overlap for cross validation and test set
        test_index = ((df["SENDER_id"].isin(all_node_list[node_test_index]) | df["RECEIVER_id"].isin(
            all_node_list[node_test_index])) & ~(
                df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
            all_node_list[node_train_index])) & ~(
                    df["SENDER_id"].isin(all_node_list[node_val_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_val_index]))).to_numpy()  # Remove overlap for cross validation and test set
        df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
        train_embeds, val_embeds, test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
        train_labels, val_labels, test_labels = labels[train_index], labels[val_index], labels[test_index]
    elif train_test_index is None:
        print("Randomly split dataset by edges...")
    return df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels, test_labels, \
           label_vectorizer, feature_vectorizer


def _sample_using_bfs(df_train, df_val, df_test, train_embeds, val_embeds, test_embeds,
                     train_labels, val_labels, test_labels, sample_size):
    print("_sample_using_bfs...")
    # df = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
    # Easier to implement:
    df = df_train.reset_index(drop=True)
    embeds = train_embeds
    labels = train_labels

    rows = df["SENDER_id"].to_numpy()
    cols = df["RECEIVER_id"].to_numpy()
    adj = scipy.sparse.coo_matrix((np.ones(rows.shape[0], int), (rows, cols)),
                                  shape=(max(rows.max()+1, cols.max()+1),
                                         max(rows.max()+1, cols.max()+1)))
    all_nodes = set()
    break_flag = False
    for i in range(adj.shape[0]):
        if i in all_nodes:
            continue
        nodes = breadth_first_order(adj, i, directed=False, return_predecessors=False)
        for node in nodes.tolist():
            all_nodes.add(node)
            if len(all_nodes) >= sample_size:
                break_flag = True
                break
        if break_flag:
            break
    index = df[df["SENDER_id"].isin(all_nodes) & df["RECEIVER_id"].isin(all_nodes)].index.to_numpy()
    train_index, test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
    train_index, val_index = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=12)
    df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]

    nodeid_dict = {}
    for df in [df_train, df_val, df_test]:
        for sndr_id, rcvr_id in zip(df["SENDER_id"], df["RECEIVER_id"]):
            if sndr_id not in nodeid_dict:
                nodeid_dict[sndr_id] = len(nodeid_dict)
            if rcvr_id not in nodeid_dict:
                nodeid_dict[rcvr_id] = len(nodeid_dict)
        df["SENDER_id"] = df["SENDER_id"].apply(lambda account, account_nodeid_dict: account_nodeid_dict[account], account_nodeid_dict=nodeid_dict)
        df["RECEIVER_id"] = df["RECEIVER_id"].apply(lambda account, account_nodeid_dict: account_nodeid_dict[account], account_nodeid_dict=nodeid_dict)

    train_embeds, val_embeds, test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
    train_labels, val_labels, test_labels = labels[train_index], labels[val_index], labels[test_index]
    return df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels, test_labels


def export_data(df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels, test_labels, label_vectorizer, feature_vectorizer, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df_train.to_csv(os.path.join(output_folder, "train_edges.csv"))
    df_val.to_csv(os.path.join(output_folder, "val_edges.csv"))
    df_test.to_csv(os.path.join(output_folder, "test_edges.csv"))
    CM.IO.ExportToPkl(os.path.join(output_folder, "train_embeds_labels.pkl"), [train_embeds, train_labels])
    CM.IO.ExportToPkl(os.path.join(output_folder, "val_embeds_labels.pkl"), [val_embeds, val_labels])
    CM.IO.ExportToPkl(os.path.join(output_folder, "test_embeds_labels.pkl"), [test_embeds, test_labels])
    CM.IO.ExportToPkl(os.path.join(output_folder, "label_vectorizer.pkl"), label_vectorizer)
    CM.IO.ExportToPkl(os.path.join(output_folder, "feature_vectorizer.pkl"), feature_vectorizer)
    print("data export", CM.Utilities.TimeElapsed(LastTime=True))
    return


def get_sampled_graph(df_train, df_val, df_test, train_embeds, val_embeds, test_embeds, train_labels, val_labels,
                      test_labels, label_vectorizer, feature_vectorizer, output_folder, sample_sizes):
    # Sample subgraphs
    for sample_size in sample_sizes:
    #     senders, receivers, reviews, labels, deduplicate_labels = sample_using_bfs(senders, receivers, reviews, labels, deduplicate_labels, sample_size=sample_size)
        df_train_sampled, df_val_sampled, df_test_sampled, train_embeds_sampled, \
        val_embeds_sampled, test_embeds_sampled, train_labels_sampled, \
        val_labels_sampled, test_labels_sampled = \
            _sample_using_bfs(df_train, df_val, df_test, train_embeds, val_embeds, test_embeds,
                           train_labels, val_labels, test_labels, sample_size)
        print(f"Sampled number of edge, train: {df_train_sampled.shape[0]}, val: {df_val_sampled.shape[0]}, "
              f"test: {df_test_sampled.shape[0]}")
        export_data(df_train_sampled, df_val_sampled, df_test_sampled, train_embeds_sampled,
                    val_embeds_sampled, test_embeds_sampled, train_labels_sampled,
                    val_labels_sampled, test_labels_sampled, label_vectorizer, feature_vectorizer,
                    output_folder + f"_bfs_sampled_size={sample_size}")
    return df_train_sampled, df_val_sampled, df_test_sampled, train_embeds_sampled, \
        val_embeds_sampled, test_embeds_sampled, train_labels_sampled, \
        val_labels_sampled, test_labels_sampled