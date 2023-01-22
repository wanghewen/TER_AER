import datetime
from copy import deepcopy, copy

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.profiler import SimpleProfiler
import os
import torch
import pytorch_lightning as pl
import CommonModules as CM

csv_logger = CSVLogger(".", name='log', version=None, prefix='')
profiler = SimpleProfiler(dirpath=".", filename="profile.log")
max_epochs = 200
log_file = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ".log"


def find_gpus(num_of_cards_needed=4):
    """
    Find the GPU which uses least memory. Should set CUDA_VISIBLE_DEVICES such that it uses all GPUs.

    :param num_of_cards_needed:
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >~/.tmp_free_gpus')
    # If there is no ~ in the path, return the path unchanged
    with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [(idx, int(x.split()[2]))
                               for idx, x in enumerate(frees)]
    idx_freeMemory_pair.sort(reverse=True)
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [idx_memory_pair[0] for idx_memory_pair in
                 idx_freeMemory_pair[:num_of_cards_needed]]
    # usingGPUs = ','.join(usingGPUs)
    print('using GPUs:', end=' ')
    for pair in idx_freeMemory_pair[:num_of_cards_needed]:
        print(f'{pair[0]} {pair[1] / 1024:.1f}GB')
    return usingGPUs


def print_result_for_pl(trainer: pl.Trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader,
                        log_file=log_file):
    last_checkpoint = checkpoint_callback.last_model_path
    print(last_checkpoint)
    print("train_dataloader, last_checkpoint", trainer.test(dataloaders=train_dataloader, ckpt_path=last_checkpoint,
                                                            verbose=False))
    print("val_dataloader, last_checkpoint", trainer.test(dataloaders=val_dataloader, ckpt_path=last_checkpoint,
                                                          verbose=False))
    print("test_dataloader, last_checkpoint", trainer.test(dataloaders=test_dataloader, ckpt_path=last_checkpoint,
                                                           verbose=False))
    print(checkpoint_callback.best_model_path)
    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)
    # trainer.validate(gcn_baseline, val_dataloaders=val_dataloader)
    # trainer.validate(dataloaders=[val_dataloader, test_dataloader], ckpt_path=best_model_path)
    train_result = trainer.test(dataloaders=train_dataloader, ckpt_path=best_model_path,
                                verbose=False)[0]
    val_result = trainer.test(dataloaders=val_dataloader, ckpt_path=best_model_path,
                              verbose=False)[0]
    start_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
    test_result = trainer.test(dataloaders=test_dataloader, ckpt_path=best_model_path,
                               verbose=False)[0]
    end_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
    print(f"Testing time elapsed: {end_testing_time - start_testing_time} sec")
    print("train_dataloader, best_model", train_result)
    print("val_dataloader, best_model", val_result)
    print("test_dataloader, best_model", test_result)
    print(f"{best_model_path},{test_result['test/accuracy_epoch']},"
          f"{test_result['test/ap_epoch']},{test_result['test/topk_precision_epoch']},"
          f"{test_result['test/macro_f1_epoch']},{test_result['test/f1_anomaly_epoch']},"
          f"{test_result['test/auc_epoch']},{test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch'])},"
          f"{test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])},{test_result['test/loss_epoch']},"
          f"{train_result['test/loss_epoch']}")
    with open(log_file, "a+") as fd:
        print(f"{best_model_path},{test_result['test/accuracy_epoch']},"
              f"{test_result['test/ap_epoch']},"
              f"{test_result['test/topk_precision_epoch']},"
              f"{test_result['test/macro_f1_epoch']}," 
              f"{test_result['test/f1_anomaly_epoch']},"
              f"{test_result['test/auc_epoch']},"
              f"{test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch'])},"
              f"{test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])},"
              f"{test_result['test/loss_epoch']},"
              f"{train_result['test/loss_epoch']}", file=fd)
    return test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch']), test_result['test/auc_epoch'], \
           test_result['test/ap_epoch'], test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])


def train(dataset_class, dataloader_class, model_class, model_parameter_dict: dict, logger, checkpoint_callback,
          use_pl, training_phase=None, previous_phase_checkpoint=None, initialize_dataset=True):
    model_parameter_dict["training_phase"] = training_phase
    if initialize_dataset:
        dataset = dataset_class()
        train_dataset, val_dataset, test_dataset = copy(dataset), copy(dataset), copy(dataset)
        # Will modify initialized_instance in __init__
        dataset_class(subset="train", initialized_instance=train_dataset)
        dataset_class(subset="val", initialized_instance=val_dataset)
        dataset_class(subset="test", initialized_instance=test_dataset)
        in_nfeats = train_dataset[0]["nfeats"].shape[1]
    else:
        train_dataset, val_dataset, test_dataset = None, None, None  # The model will use dataset class to get instances
        in_nfeats = None
    # from sklearn.metrics import roc_auc_score
    # print(roc_auc_score(train_dataset[0]["labels"][train_dataset[0]["nodes_mask"]].numpy(),
    #               train_dataset[0]["nfeats"][train_dataset[0]["nodes_mask"]][:, -1].numpy()))
    # print(roc_auc_score(val_dataset[0]["labels"][val_dataset[0]["nodes_mask"]].numpy(),
    #               val_dataset[0]["nfeats"][val_dataset[0]["nodes_mask"]][:, -1].numpy()))
    # print(roc_auc_score(test_dataset[0]["labels"][test_dataset[0]["nodes_mask"]].numpy(),
    #               test_dataset[0]["nfeats"][test_dataset[0]["nodes_mask"]][:, -1].numpy()))
    # return None, None, None, None, None
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if (torch.cuda.is_available() and not (model_parameter_dict.get("force_cpu", False)))\
            and (os.environ["CUDA_VISIBLE_DEVICES"] not in ["-1", ""]):
        # gpus = -1
        gpus = find_gpus(1)
        accelerator = "gpu"
    else:
        gpus = None
        accelerator = "cpu"
    if dataloader_class is not None:  # In case you don't use dataloaders in your fitting code
        train_dataloader = dataloader_class(device=gpus).build_dataloader(train_dataset, train_val_test="train")
        val_dataloader = dataloader_class(device=gpus).build_dataloader(val_dataset, train_val_test="val")
        test_dataloader = dataloader_class(device=gpus).build_dataloader(test_dataset, train_val_test="test")
    else:
        train_dataloader, val_dataloader, test_dataloader = None, None, None
    if model_parameter_dict.get("n_classes", None) == "auto":  # Infer n_classes during run time
        if len(train_dataset.train_labels.shape) > 1:
            n_classes = train_dataset.train_labels.shape[1]
        else:
            n_classes = 1
        model_parameter_dict["n_classes"] = n_classes  # Slightly conflict with labelsize. Should fix in the future
    try:  # For edge attributed models
        if previous_phase_checkpoint:
            model = model_class.load_from_checkpoint(previous_phase_checkpoint, in_nfeats=in_nfeats,
                                                     in_efeats=in_nfeats, labelsize=1,
                                                     **model_parameter_dict)
        else:
            model = model_class(in_nfeats=in_nfeats, in_efeats=in_nfeats, labelsize=1, **model_parameter_dict)
    except TypeError:
        if previous_phase_checkpoint:
            model = model_class.load_from_checkpoint(previous_phase_checkpoint, in_nfeats=in_nfeats,
                                                     labelsize=1, **model_parameter_dict)
        else:
            model = model_class(in_nfeats=in_nfeats, labelsize=1, **model_parameter_dict)
    # if previous_phase_checkpoint:
    #     model: BaseModel
    #     model = model.load_from_checkpoint(previous_phase_checkpoint)
    if use_pl:
        if training_phase != "discrimination":
            earlystopping_callback = EarlyStopping(monitor='val/loss_epoch',
                                                   min_delta=0.00,
                                                   # patience=1,
                                                   patience=10,
                                                   verbose=True,
                                                   mode='min')
            callbacks = [checkpoint_callback, earlystopping_callback]
        else:
            callbacks = [checkpoint_callback]
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=gpus,
                             callbacks=callbacks,
                             logger=[logger, csv_logger], check_val_every_n_epoch=10, profiler=False,
                             move_metrics_to_cpu=True, precision=32)
        # Because there may be another call inside fit, we define these variables to record time
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        # if model_parameter_dict["structure"] == "GAT":
        #     import ipdb; ipdb.set_trace()
        trainer.fit(model, train_dataloader, [val_dataloader, test_dataloader])
        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        if training_phase:
            print(f"Training time elapsed for phase {training_phase}: {end_training_time - start_training_time} sec")
        else:
            print(f"Training time elapsed: {end_training_time - start_training_time} sec")
        print(checkpoint_callback.best_model_path)
        return trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader, None, model
    else:  # Let the model handle all the training
        test_results = model.fit([train_dataloader, val_dataloader, test_dataloader], max_epochs=max_epochs,
                                 gpus=gpus, dataset_class=dataset_class)
        return model, None, None, None, None, test_results, model


def test(trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader, use_pl):
    test_results = print_result_for_pl(trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader)
    return test_results


def main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl=True,
         initialize_dataset=True):
    """

    :param logger_name:
    :param dataset_class:
    :param dataloader_class:
    :param model_class:
    :param model_parameter_dict:
    :param use_pl:
    :param initialize_dataset: Whether automatically initialize dataset or let the model handle the dataset class  instantiation
    :return:
    """
    pl.seed_everything(12)
    logger = TensorBoardLogger("logs_20220905", name=logger_name, version=None)
    # logger = TensorBoardLogger("logs_20220831", name=logger_name, version=None)
    previous_phase_checkpoint = None
    if "training_phases" in model_parameter_dict:
        training_phases = model_parameter_dict["training_phases"]  # e.g. ['discrimination', 'classification']
    else:
        training_phases = [None]
    for training_phase_index, training_phase in enumerate(training_phases):
        if training_phase_index >= 1:
            model_parameter_dict["training_phase"] = training_phase
            previous_phase_checkpoint = checkpoint_callback.best_model_path
            # model_class_instance = model_class_instance.load_from_checkpoint(checkpoint_callback.best_model_path,
            #                                                                  **model_parameter_dict)
        if training_phase == "discrimination":
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./models", logger.name, str(datetime.datetime.now())),
                filename='epoch={epoch}-train_loss_epoch={'
                         'train/loss_epoch:.6f}',
                # monitor='val/best_macro_f1_epoch', mode="max", save_last=True,
                # monitor='val/auc_epoch', mode="max", save_last=True,
                # monitor='val/macro_f1_epoch', mode="max", save_last=True,
                # monitor='val/ap_epoch', mode="max", save_last=True,
                monitor='train/loss_epoch', mode="min", save_last=True,
                auto_insert_metric_name=False)
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./models", logger.name, str(datetime.datetime.now())),
                filename='epoch={epoch}-val_loss_epoch={'
                         'val/loss_epoch:.6f}-val_accuracy_epoch={'
                         'val/accuracy_epoch:.4f}-val_macro_f1_epoch={'
                         'val/macro_f1_epoch:.4f}-val_topk_precision_epoch={'
                         'val/topk_precision_epoch:.4f}-val_f1_anomaly_epoch={'
                         'val/f1_anomaly_epoch:.4f}',
                # monitor='val/best_macro_f1_epoch', mode="max", save_last=True,
                monitor='val/auc_epoch', mode="max", save_last=True,
                # monitor='val/macro_f1_epoch', mode="max", save_last=True,
                # monitor='val/ap_epoch', mode="max", save_last=True,
                # monitor='val/loss_epoch', mode="min", save_last=True,
                auto_insert_metric_name=False)
        trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader, \
        test_results, model_class_instance = \
            train(dataset_class, dataloader_class, model_class, model_parameter_dict, logger, checkpoint_callback,
                  use_pl, training_phase=training_phase, previous_phase_checkpoint=previous_phase_checkpoint,
                  initialize_dataset=initialize_dataset)
    # return
    if use_pl:
        test_results = test(trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader, use_pl)
    return test_results


