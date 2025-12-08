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


    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    os.makedirs(os.path.join(save_path, experiment_name), exist_ok=True)
    plt.savefig(os.path.join(save_path, experiment_name, f'{figname}.png'))
    plt.close()

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

    return metrics, cm


def save_history_results(history: Any, cfg: dict, dpi: int = 96):
    """Save FL history results (distributed and centralized) as plots (PNG), CSV and JSON.

    - Plots: loss and accuracy over communication rounds (PNG)
    - CSV/JSON: distributed metrics per round and centralized test metrics per round

    Files are written to `os.path.join(save_path, experiment_name)`; if `save_path` is
    not present in `cfg`, defaults to `results/`.
    """
    save_base = cfg.get("save_path", "results") or "results"
    experiment_name = cfg.get("experiment_name", "")
    out_dir = os.path.join(save_base, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    # Print summary (match user's snippet)
    try:
        log.logger.info("Output FL - Loss: %s", getattr(history, "losses_distributed", None))
        log.logger.info("Output FL - Accuracy: %s", getattr(history, "metrics_distributed", None))

        log.logger.info("Centralized metrics: %s", getattr(history, "metrics_centralized", None))
        log.logger.info("Centralized losses: %s", getattr(history, "losses_centralized", None))
    except Exception:
        pass

    # Distributed losses
    fl_loss = []
    fl_loss_rounds = []
    for loss in getattr(history, "losses_distributed", []) or []:
        try:
            fl_loss_rounds.append(int(loss[0]))
            fl_loss.append(float(loss[1]))
        except Exception:
            continue

    # Distributed metrics (e.g., accuracy)
    fl_accuracy = []
    fl_accuracy_rounds = []
    metrics_distributed = getattr(history, "metrics_distributed", {}) or {}
    if "accuracy" in metrics_distributed:
        for acc in metrics_distributed["accuracy"]:
            try:
                fl_accuracy_rounds.append(int(acc[0]))
                fl_accuracy.append(float(acc[1]))
            except Exception:
                continue

    # Plot Loss
    communication_round = range(len(fl_loss))
    plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(communication_round, fl_loss, linewidth=1, linestyle="solid", marker=".", color="black")
    plt.xlabel("Communication Round(#)", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.grid(linestyle=":", linewidth="0.5")
    loss_png = os.path.join(out_dir, "fl_loss.png")
    plt.savefig(loss_png)
    plt.close()

    # Plot Accuracy
    communication_round = range(len(fl_accuracy))
    plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(communication_round, fl_accuracy, linewidth=1, linestyle="solid", marker=".", color="black")
    plt.xlabel("Communication Round(#)", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.grid(linestyle=":", linewidth="0.5")
    acc_png = os.path.join(out_dir, "fl_accuracy.png")
    plt.savefig(acc_png)
    plt.close()

    # Build distributed per-round table
    rounds_data = {}
    # losses
    for r, v in getattr(history, "losses_distributed", []) or []:
        try:
            rr = int(r)
        except Exception:
            continue
        rounds_data.setdefault(rr, {})
        rounds_data[rr]["loss"] = float(v)

    # metrics (multiple metrics possible)
    for metric_name, metric_list in (metrics_distributed or {}).items():
        for r, v in metric_list:
            try:
                rr = int(r)
            except Exception:
                continue
            rounds_data.setdefault(rr, {})
            try:
                rounds_data[rr][metric_name] = float(v)
            except Exception:
                rounds_data[rr][metric_name] = v

    # Write distributed CSV
    if rounds_data:
        fieldnames = ["round"] + sorted({k for d in rounds_data.values() for k in d.keys()})
        csv_path = os.path.join(out_dir, "distributed_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rr in sorted(rounds_data.keys()):
                row = {"round": rr}
                row.update(rounds_data[rr])
                writer.writerow(row)

        # JSON
        with open(os.path.join(out_dir, "distributed_metrics.json"), "w", encoding="utf-8") as jf:
            json.dump({str(k): v for k, v in rounds_data.items()}, jf, indent=2)

    # Save centralized test metrics/losses if present
    central_metrics = getattr(history, "metrics_centralized", {}) or {}
    central_losses = getattr(history, "losses_centralized", []) or []
    central_data = {}
    # central metrics: dict metric -> list[(round, value), ...]
    for metric_name, metric_list in central_metrics.items():
        for r, v in metric_list:
            try:
                rr = int(r)
            except Exception:
                continue
            central_data.setdefault(rr, {})
            try:
                central_data[rr][metric_name] = float(v)
            except Exception:
                central_data[rr][metric_name] = v

    # central losses: list[(round, loss), ...]
    for r, v in central_losses:
        try:
            rr = int(r)
        except Exception:
            continue
        central_data.setdefault(rr, {})
        central_data[rr]["loss"] = float(v)

    if central_data:
        fieldnames = ["round"] + sorted({k for d in central_data.values() for k in d.keys()})
        csv_path = os.path.join(out_dir, "centralized_test_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rr in sorted(central_data.keys()):
                row = {"round": rr}
                row.update(central_data[rr])
                writer.writerow(row)

        # JSON
        with open(os.path.join(out_dir, "centralized_test_metrics.json"), "w", encoding="utf-8") as jf:
            json.dump({str(k): v for k, v in central_data.items()}, jf, indent=2)

    log.logger.info(f"Saved history plots and metrics to {out_dir}")