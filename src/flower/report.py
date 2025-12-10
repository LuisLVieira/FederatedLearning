import os
import log
from models.models_definition import build_model
from models.model_train import evaluate
from torch import load
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import csv
from typing import Any

def plot_confusion_matrices(cm, class_names, save_path: str = '.', experiment_name: str = '', figname: str = 'confusion_matrix'):

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    os.makedirs(os.path.join(save_path, experiment_name), exist_ok=True)

    fig.savefig(os.path.join(save_path, experiment_name, f'{figname}.png'))

    return fig


def best_model_test(history, cfg, _device, class_to_target, testloader, dataset):
    model_config = cfg.get("model_config", {})
    save_path = os.path.join(cfg.get("save_path", ""), cfg.get("experiment_name", ""), "models")

    round_files = [f for f in os.listdir(save_path) if f.startswith("model_round")]

    if len(round_files) == 0:
        raise RuntimeError(f"Nenhum modelo encontrado em {save_path}")

    # extrai números dos rounds
    round_numbers = sorted([
        int(f.replace("model_round", "").replace(".pt", ""))
        for f in round_files
    ])

    try:
        # history.losses_distributed = [(round, loss), ...]
        losses = history.losses_distributed
        best_round = min(losses, key=lambda x: x[1])[0]
    except:
        # fallback: usa último round
        best_round = round_numbers[-1]

    best_model_path = os.path.join(save_path, f"model_round{best_round}.pt")
    log.logger.info(f"\n Carregando melhor modelo encontrado: Round {best_round}")
    log.logger.info(f"Path: {best_model_path}")

    # ---------------------------------------------------------
    # CARREGAR O MODELO GLOBAL
    # ---------------------------------------------------------

    global_model = build_model(
        model_name=model_config.get("model", "custom_layer4_fc_resnet18"),
        num_classes=len(class_to_target)
    )
    global_model.load_state_dict(load(best_model_path, map_location=_device))
    global_model.to(_device)
    global_model.eval()

    # ---------------------------------------------------------
    # AVALIAÇÃO GLOBAL
    # ---------------------------------------------------------

    metrics, cm = evaluate(global_model, dataset, testloader, _device, len(class_to_target), model_config)

    log.logger.info("\n Métricas do Melhor Modelo Global:")
    log.logger.info(metrics)

    return global_model, metrics, cm 


def save_history_results(history: dict, cfg: dict, dpi: int = 96):
    """
    Save FL history results:
    - aggregated metrics
    - client metrics
    - plots
    Returns dicts + matplotlib figure (combined accuracy/loss).
    """
    import json, os
    import matplotlib.pyplot as plt

    save_base = cfg.get("save_path", "results")
    experiment_name = cfg.get("experiment_name", "")
    out_dir = os.path.join(save_base, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    central_metrics = history.get("metrics_centralized", {})
    central_losses = history.get("losses_centralized", [])
    distributed_metrics = history.get("metrics_distributed", {})
    distributed_losses = history.get("losses_distributed", [])
    client_metrics = history.get("clients", {})

    # === Build Distributed ===
    distributed_data = {}
    for round, loss in distributed_losses:
        distributed_data.setdefault(round, {})
        distributed_data[round]["loss"] = float(loss)

    for metric_name, metric_list in distributed_metrics.items():
        for round, value in metric_list:
            distributed_data.setdefault(round, {})
            try:
                distributed_data[round][metric_name] = float(value)
            except Exception:
                distributed_data[round][metric_name] = value

    # === Build Centralized ===
    central_data = {}
    for round, loss in central_losses:
        central_data.setdefault(round, {})
        central_data[round]["loss"] = float(loss)

    for metric_name, metric_list in central_metrics.items():
        for round, value in metric_list:
            central_data.setdefault(round, {})
            try:
                central_data[round][metric_name] = float(value)
            except Exception:
                central_data[round][metric_name] = value

    # === Save JSON ===
    if distributed_data:
        with open(os.path.join(out_dir, "distributed_metrics.json"), "w") as jf:
            json.dump({str(k): v for k, v in distributed_data.items()}, jf, indent=2)

    if central_data:
        with open(os.path.join(out_dir, "centralized_test_metrics.json"), "w") as jf:
            json.dump({str(k): v for k, v in central_data.items()}, jf, indent=2)

    # === Save Client Metrics ===
    if client_metrics:
        clients_dir = os.path.join(out_dir, "clients")
        os.makedirs(clients_dir, exist_ok=True)

        for cid, round_metrics in client_metrics.items():
            cfile = os.path.join(clients_dir, f"client_{cid}.json")
            out = {}
            if os.path.exists(cfile):
                out = json.load(open(cfile))

            for round, metrics in round_metrics.items():
                out[str(round)] = metrics

            with open(cfile, "w") as jf:
                json.dump(out, jf, indent=2)

    # === Build Plot ===
    # prepare series
    fl_loss = [v["loss"] for k, v in distributed_data.items()]
    fl_acc = [v.get("accuracy") for k, v in distributed_data.items()]

    rounds_loss = list(range(len(fl_loss)))
    rounds_acc = list(range(len(fl_acc)))

    # joint fig
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    axes[0].plot(rounds_loss, fl_loss, marker=".")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Round")

    axes[1].plot(rounds_acc, fl_acc, marker=".")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Round")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distributed_metrics.png"))

    return distributed_data, central_data, client_metrics, fig
