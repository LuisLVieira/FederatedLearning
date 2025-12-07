from flwr.server.strategy import FedProx, FedAvg, FedAdagrad, FedAdam, FedYogi, Krum, DPFedAvgAdaptive, QFedAvg, FaultTolerantFedAvg
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

def get_fedadagrad(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
    return FedAdagrad(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        eta=cfg.get("eta", 0.1),
        tau=cfg.get("tau", 1e-9),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_fedadam(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
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
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_fedyogi(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
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
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_krum(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
    return Krum(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        krum_config=cfg.get("krum_config", {"multi_krum": False, "num_krum": 1}),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_dp_fedavg_adaptive(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
    return DPFedAvgAdaptive(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        noise_multiplier=cfg.get("noise_multiplier", 1.0),
        l2_norm_clip=cfg.get("l2_norm_clip", 1.0),
        adaptive_clip=cfg.get("adaptive_clip", True),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

def get_qfedavg(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
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

def get_faulttolerant_fedavg(save_path, num_classes, testloader, device, model_name, model_config, **cfg):
    return FaultTolerantFedAvg(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=cfg.get("num_clients", 7),
        min_evaluate_clients=cfg.get("num_clients", 7),
        min_available_clients=cfg.get("num_clients", 7),
        evaluate_fn=server_evaluate_fn(num_classes, testloader, device, model_name=model_name, model_config=model_config, save_path=save_path),
        evaluate_metrics_aggregation_fn=metrics_agg,
    )