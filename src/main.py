import json
import argparse
import log
import pandas as pd
from data_processing.dataset import (
    split_on_train_test,
    show_train_test_info,
    split_on_non_iid_clients,
    plot_class_distribution_on_clients
)
from models import models_definition
from data_processing.dataloader import load_datasets, plot_load_data
import numpy as np
import random
from torch import manual_seed, device, cuda, load
import os
from flower.report import best_model_test, plot_confusion_matrices, save_history_results
from flower.simulation import simulation


def main():
    # load data and configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    
    if ("save_path" in cfg) and (cfg["save_path"]):
        os.makedirs(os.path.join(cfg["save_path"], cfg.get("experiment_name", "")), exist_ok=True)

    random.seed(cfg.get("random_seed", 42))
    np.random.seed(cfg.get("random_seed", 42))
    manual_seed(cfg.get("random_seed", 42))

    log.logger.info(f'Config loaded: {cfg}\n')

    dataset = pd.read_csv(cfg["metadata_path"])

    # trick to find path correctly
    dataset["path"] = dataset["path"].str.replace("./data", cfg.get("dataset_path", "./data"), regex=False)

    log.logger.info(f'Dataset loaded with shape: {dataset.shape}')
    log.logger.info(f'Dataset loaded with columns: {dataset.columns.to_list()}\n')

    target_to_class = dict(zip(dataset["target"], dataset["Class"]))
    class_to_target = dict(zip(dataset["Class"], dataset["target"]))

    log.logger.info(f'label mapping: {class_to_target}\n')

    # split data on train and test
    clients_data, val_data, test_data = split_on_train_test(
        data=dataset,
        test_ratio=cfg.get("test_ratio", 0.2),
        val_ratio=cfg.get("global_valid_ratio", 0),
        random_seed=cfg.get("random_seed", 42)
    )

    show_train_test_info(
        clients_data,
        val_data,
        test_data,
        save_path=cfg.get("save_path", ""),
        experiment_name=cfg.get("experiment_name", ""),
        plot=False
    )

    # get client data from train set

    clients = split_on_non_iid_clients(
        clients_df=clients_data,
        num_clients=cfg.get("num_clients", 10),
        val_frac=cfg.get("val_frac", 0.15),
        random_seed=cfg.get("random_seed", 42),
        alpha=cfg.get("alpha", 1.0)
    )

    plot_class_distribution_on_clients(
        clients,
        num_clients=cfg.get("num_clients", 10),
        save_path=cfg.get("save_path", ""),
        experiment_name=cfg.get("experiment_name", ""),
        figname="class_distribution",
        plot=False
    )

    # data cache
    trainloaders, valloaders, globalvalloader, testloader = load_datasets(
        clients=clients,
        val_df=val_data,
        test_df=test_data,
        transforms_config=cfg["data_transform"],
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 0),
        label_map=class_to_target
    )

    # Not necessary step
    # plot_load_data(
    #     trainloaders,
    #     valloaders,
    #     cfg.get("num_clients", 10),
    #     label_map=class_to_target,
    #     save_path=cfg.get("save_path", ""),
    #     experiment_name=cfg.get("experiment_name", ""),
    #     plot=False,
    #     figname="original_class_distribution"
    # )

    # Model training and evaluation here

    model_config = cfg.get("model_config", {})

    with open(os.path.join(cfg.get("save_path", ""), cfg.get("experiment_name", ""), "config.json"), 'w') as f:
        json.dump(cfg, f, indent=4)

    model = models_definition.build_model(
        model_name=model_config.get("model", "custom_layer4_fc_resnet18"),
        num_classes=len(class_to_target)
    )

    log.logger.info(f'Model: {model}\n')

    # Check trainable and not trainable layers
    trainable_layers = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)

    log.logger.info("Trainable layers:")
    for name in trainable_layers:
        print(name)

    # Federated Learning process here ...
    _device = device("cuda" if cuda.is_available() else "cpu")

    history = simulation(
        trainloaders,
        valloaders,
        globalvalloader,
        device=_device,
        dataset=dataset,
        num_classes=len(class_to_target),
        model=model,
        **cfg
    )

    log.logger.info(f"Training history: {history}")

    # Save training history

    save_history_results(history, cfg, dpi=96)

    # Best model test

    val_metrics, val_cm = best_model_test(history, cfg, _device, class_to_target, globalvalloader, dataset)

    plot_confusion_matrices(
        val_cm,
        class_names=[target_to_class[i] for i in range(len(class_to_target))],
        save_path=cfg.get("save_path", ""),
        experiment_name=cfg.get("experiment_name", ""),
        figname="confusion_matrix_best_model"
    )

    metrics_df = pd.DataFrame([val_metrics])
    metrics_df.to_csv(os.path.join(cfg.get("save_path", ""), cfg.get("experiment_name", ""), "global_val_metrics.csv"), index=False)

    test_metrics, test_cm = best_model_test(history, cfg, _device, class_to_target, testloader, dataset)

    plot_confusion_matrices(
        test_cm,
        class_names=[target_to_class[i] for i in range(len(class_to_target))],
        save_path=cfg.get("save_path", ""),
        experiment_name=cfg.get("experiment_name", ""),
        figname="confusion_matrix_best_model_test"
    )

    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(os.path.join(cfg.get("save_path", ""), cfg.get("experiment_name", ""), "test_metrics.csv"), index=False)


if __name__ == "__main__":
    main()