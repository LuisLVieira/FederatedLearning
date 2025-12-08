from flwr.server.strategy import FedProx, FedAvg, FedAdagrad, FedAdam, FedYogi, Krum, DPFedAvgAdaptive, QFedAvg, FaultTolerantFedAvg
import flwr as fl
from .server import server_evaluate_fn
from flwr.common import ndarrays_to_parameters
from torch import nn
from typing import List
import numpy as np

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def metrics_agg(metrics):
    """Aggregate metrics from all clients using weighted average.
    
    Args:
        metrics: list of (num_examples, {metric_dict}) tuples from each client
        
    Returns:
        dict: Aggregated metrics weighted by number of examples per client
    """
    if not metrics:
        return {}
    
    # Collect all unique metric keys across all clients
    all_keys = set()
    for _, metric_dict in metrics:
        all_keys.update(metric_dict.keys())
    
    total_examples = sum(num for num, _ in metrics)
    if total_examples == 0:
        return {}
    
    aggregated = {}
    for key in all_keys:
        # Only aggregate metrics present in all clients, or use 0 for missing
        values = []
        for num, m in metrics:
            if key in m:
                values.append(num * m[key])
            else:
                # If a metric is missing, it counts as 0 (or skip it)
                values.append(0)
        aggregated[key] = sum(values) / total_examples
    
    return aggregated


def fit_config(server_round: int, model_config):
    return {
        "round": server_round,
        "local_epochs": model_config["epochs"],
    }

def eval_config(server_round: int):
    return {
        "round": server_round
    }

def get_fedprox(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    return FedProx(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        proximal_mu=cfg.get("mu", 0.05),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )


def get_fedavg(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    return FedAvg(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )


def get_fedadagrad(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    return FedAdagrad(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        initial_parameters=initial_parameters,
        eta=cfg.get("eta", 0.1),
        tau=cfg.get("tau", 1e-9),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_fedadam(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    return FedAdam(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        eta=cfg.get("eta", 0.001),
        eta_l=cfg.get("eta_l", 0.001),
        beta_1=cfg.get("beta_1", 0.9),
        beta_2=cfg.get("beta_2", 0.99),
        tau=cfg.get("tau", 1e-9),
        initial_parameters=initial_parameters,
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_fedyogi(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    return FedYogi(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        eta=cfg.get("eta", 0.001),
        eta_l=cfg.get("eta_l", 0.001),
        beta_1=cfg.get("beta_1", 0.9),
        beta_2=cfg.get("beta_2", 0.99),
        tau=cfg.get("tau", 1e-9),
        initial_parameters=initial_parameters,
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_krum(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    return Krum(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        krum_config=cfg.get("krum_config", {"multi_krum": False, "num_krum": 1}),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_dp_fedavg_adaptive(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    return DPFedAvgAdaptive(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        noise_multiplier=cfg.get("noise_multiplier", 1.0),
        l2_norm_clip=cfg.get("l2_norm_clip", 1.0),
        adaptive_clip=cfg.get("adaptive_clip", True),
        initial_parameters=initial_parameters,
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_qfedavg(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    return QFedAvg(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        q_param=cfg.get("q_param", 0.5),  # ajustar conforme necessário
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_faulttolerant_fedavg(save_path, num_classes, testloader, device, model_name, model_config, model, **cfg):
    return FaultTolerantFedAvg(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )
