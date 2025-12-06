from flwr.server.strategy import FedProx
from flwr.server.strategy import FedAvg
import flwr as fl
from .server import server_evaluate_fn


def metrics_agg(metrics):
    # metrics = [(num_examples, {"loss": ..., "accuracy": ...}), ...]
    total_examples = sum(num for num, _ in metrics)
    aggregated = {}
    for key in metrics[0][1].keys():
        aggregated[key] = sum(num * m[key] for num, m in metrics) / total_examples
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

def get_fedprox(save_path, num_classes, testloader, device, model_name, model_config, **fedprox_cfg):
    return FedProx(
        fraction_fit=fedprox_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=fedprox_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=fedprox_cfg.get("num_clients", 7),
        min_evaluate_clients=fedprox_cfg.get("num_clients", 7),
        min_available_clients=fedprox_cfg.get("num_clients", 7),
        proximal_mu=fedprox_cfg.get("mu", 0.05),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )


def get_fedavg(save_path, num_classes, testloader, device, model_name, model_config, **fedavg_cfg):
    return FedAvg(
        fraction_fit=fedavg_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=fedavg_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=fedavg_cfg.get("num_clients", 7),
        min_evaluate_clients=fedavg_cfg.get("num_clients", 7),
        min_available_clients=fedavg_cfg.get("num_clients", 7),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )