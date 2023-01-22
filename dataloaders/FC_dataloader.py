import scipy.sparse
import torch
from torch.utils.data import DataLoader, TensorDataset
# from global_parameters import data_folder
from icecream import ic


class FCDataloader():
    @staticmethod
    def build_dataloader(dataset, batch_size=None, train_val_test=None, device="cpu"):
        dataset = dataset[0]
        embeds, labels, nodes_mask = dataset["nfeats"], dataset["labels"], dataset["nodes_mask"]
        # train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "train_embeds_labels.pkl"))
        # val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "val_embeds_labels.pkl"))
        # test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "test_embeds_labels.pkl"))
        # embeds, labels = embeds[nodes_mask].to(device), labels[nodes_mask].to(device)
        embeds, labels = embeds[nodes_mask], labels[nodes_mask]
        ic(embeds.shape, labels.shape)
        # embeds = embeds.astype(np.float32)
        labels[labels > 0] = 1
        dataset = TensorDataset(embeds, labels)

        if batch_size is None:
            batch_size = len(dataset)
            if train_val_test == "train":
                num_workers = 10
                persistent_workers = True
            else:
                num_workers = 0
                persistent_workers = False
            dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                                    num_workers=num_workers, shuffle=False, persistent_workers=persistent_workers)

        return dataloader

    @staticmethod
    def build_dataloader_non_graph(embeds, labels, batch_size=None, shuffle=False, train_val_test=None):
        ic(embeds.shape, labels.shape)
        # embeds = embeds.astype(np.float32)
        labels[labels > 0] = 1
        labels = labels.long()
        dataset = TensorDataset(embeds, labels)

        if batch_size is None:
            batch_size = len(dataset)

        if train_val_test == "train":
            num_workers = 20
            prefetch_factor = 20
            persistent_workers = True
        else:
            num_workers = 0
            prefetch_factor = 2
            persistent_workers = False
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, prefetch_factor=prefetch_factor,
                                num_workers=num_workers, shuffle=shuffle, persistent_workers=persistent_workers)

        return dataloader
