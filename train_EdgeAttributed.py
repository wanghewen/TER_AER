import argparse
import os
import traceback
from functools import partial, update_wrapper

import torch

from dataloaders.GNN_dataloader import GNNDataloader
from models.GNNVanilla import GNNVanilla
from train import main


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--dataset', type=str, help='dataset class name', default=None)
parser.add_argument('--data_folder', type=str, help='folder where your data files stored', default=None)
args = parser.parse_args()
data_folder = args.data_folder


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    # Set class function to partial_func
    for method_name in dir(func):
        if callable(getattr(func, method_name)) and not method_name.startswith("__"):
            setattr(partial_func, method_name, getattr(func, method_name))
    return partial_func


def run_vanilla_experiments_edge_attributed(dataset_class, sample_size):
    print("Current dataset:", dataset_class.__name__)

    logger_name = dataset_class.__name__ + "__MLP__baseline"
    use_pl = False
    for svd_dim in [256]:
        model_parameter_dict = {"n_hidden": 128,
                                "n_classes": "auto",
                                "n_layers": 2,
                                "activation": torch.nn.ReLU(),
                                "dropout": 0.5,
                                # "dropout": 1,
                                "structure": "GCN",
                                "edge_attributed": True,
                                "svd_dim": svd_dim,
                                "use_gb": False,
                                "use_random_feature": False,
                                "train_ratio": 1.0,
                                "classifier": "mlp"}
        if dataset_class in [DBLPEdgeAttributed]:
            if sample_size is None:
                model_parameter_dict["train_ratio"] = 0.1  # Reduce training size

        dataloader_class = GNNDataloader
        model_class = GNNVanilla
        default_parameter_dict = model_parameter_dict | {"use_epp": False, "use_ter": False, "use_lra": False,
                                                         "use_aer": False, "use_ppr": False, "use_hkpr": False,
                                                         "use_edge2vec": False, "error_analysis": False}


        print("######################### Only TER_PPR ##########################")
        update_parameter_dict = {"use_epp": False,  "use_ter": True, "use_lra": False, "use_ppr": True,
                                 "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__+f"_{sample_size}")}
        model_parameter_dict |= default_parameter_dict | update_parameter_dict
        main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        print("######################### Only TER_HKPR ##########################")
        update_parameter_dict = {"use_epp": False,  "use_ter": True, "use_lra": False, "use_hkpr": True,
                                 "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__+f"_{sample_size}")}
        model_parameter_dict |= default_parameter_dict | update_parameter_dict
        main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        print("######################### Only AER_PPR #########################")
        update_parameter_dict = {"use_aer": True, "use_ppr": True,
                                 "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__+f"_{sample_size}")}
        model_parameter_dict |= default_parameter_dict | update_parameter_dict
        main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        print("######################### Only AER_HKPR #########################")
        update_parameter_dict = {"use_aer": True, "use_hkpr": True,
                                 "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__+f"_{sample_size}")}
        model_parameter_dict |= default_parameter_dict | update_parameter_dict
        main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        print("######################### AER_PPR + TER #########################")
        ppr_decay_factor = 0.5
        error_threshold = 1e-5
        svd_dim = 256
        # for error_threshold in [1, 1e-1, 1e-3, 1e-5, 1e-7]:
        for error_threshold in [1e-5]:
        # for ppr_decay_factor in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # for error_threshold in [0.00001]:
        # for svd_dim in [16, 32, 64, 128, 256]:
            print(f"ppr_decay_factor={ppr_decay_factor}, error_threshold={error_threshold}, svd_dim={svd_dim}")
            update_parameter_dict = {"use_ter": True, "use_aer": True, "use_ppr": True, "ppr_decay_factor":
                                     ppr_decay_factor, "error_threshold": error_threshold,
                                     "svd_dim": svd_dim,
                                     "cache_folder_path": os.path.join(data_folder, "cache",
                                                                       dataset_class.__name__+f"_{sample_size}"),
                                     # "cache_folder_path": None
                                     }
            model_parameter_dict |= default_parameter_dict | update_parameter_dict
            main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        print("######################### AER_HKPR + TER #########################")
        hkpr_heat_constant = 1
        error_threshold = 1e-5
        # for error_threshold in [1, 1e-1, 1e-3, 1e-5, 1e-7]:
        # for hkpr_heat_constant in [1, 2, 3, 4, 5]:
        for error_threshold in [0.00001]:
            print(f"hkpr_heat_constant={hkpr_heat_constant}, error_threshold={error_threshold}")
            update_parameter_dict = {"use_ter": True, "use_aer": True, "use_hkpr": True,
                                     "hkpr_heat_constant": hkpr_heat_constant,
                                     "error_threshold": error_threshold,
                                     "cache_folder_path": os.path.join(data_folder, "cache",
                                                                       dataset_class.__name__+f"_{sample_size}")}
            model_parameter_dict |= default_parameter_dict | update_parameter_dict
            main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)



if __name__ == "__main__":
    ######################Use EPP with FC#########################
    from datasets.Citations.AMinerEdgeAttributed import AMinerEdgeAttributed
    from datasets.Citations.DBLPEdgeAttributed import DBLPEdgeAttributed
    from datasets.Citations.MINDEdgeAttributed import MINDEdgeAttributed
    from datasets.Citations.OAGEdgeAttributed import OAGEdgeAttributed
    from datasets.Mooc.MoocEdgeAttributed import MoocEdgeAttributed
    from datasets.Reddit.RedditEdgeAttributed import RedditEdgeAttributed
    from datasets.Yelpchi.YelpchiEdgeAttributed import YelpchiEdgeAttributed

    if args.dataset is not None:
        dataset_classes = [eval(args.dataset)]
    else:
        dataset_classes = [YelpchiEdgeAttributed, DBLPEdgeAttributed,
                          RedditEdgeAttributed, MoocEdgeAttributed, OAGEdgeAttributed,
                          AMinerEdgeAttributed, MINDEdgeAttributed]
    # run_vanilla_experiments_edge_attributed = fuckit(run_vanilla_experiments_edge_attributed)
    for dataset_class in dataset_classes:
        try:
            print("dataset_class", dataset_class.__name__)
            if dataset_class in [OAGEdgeAttributed, AMinerEdgeAttributed]:
                for sample_size in [40000]:
                    new_dataset_class = wrapped_partial(dataset_class, add_self_loop=False, randomize_train_test=False,
                                                        randomize_by_node=False, sample_size=sample_size,
                                                        reduced_dim=256)
                    # For citations, already split when preprocessing
                    run_vanilla_experiments_edge_attributed(new_dataset_class, sample_size)
            elif dataset_class in [DBLPEdgeAttributed, MINDEdgeAttributed]:
                for sample_size in [None]:
                    new_dataset_class = wrapped_partial(dataset_class, add_self_loop=False, randomize_train_test=False,
                                                        randomize_by_node=False, sample_size=sample_size,
                                                        reduced_dim=256)
                    # For citations, already split when preprocessing
                    run_vanilla_experiments_edge_attributed(new_dataset_class, sample_size)
            else:
                new_dataset_class = wrapped_partial(dataset_class, add_self_loop=False, randomize_train_test=True,
                                                    randomize_by_node=False)
                run_vanilla_experiments_edge_attributed(new_dataset_class, None)
        except Exception as e:
            traceback.print_exc()

