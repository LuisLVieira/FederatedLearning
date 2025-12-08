from flwr.server.strategy import FedProx
from flwr.server.strategy import FedAvg
import flwr as fl
from .server import server_evaluate_fn


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

def get_fedprox(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
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


def get_fedavg(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
    return FedAvg(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )