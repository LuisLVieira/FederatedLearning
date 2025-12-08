import os
from flwr.common import Context
from . import aggregation as agg
from .clients import client_fn as original_client_fn

os.environ["FLWR_SIMULATION_USE_RAY"] = "0"
os.environ["FLWR_LOGGING"] = "error"

import flwr as fl

# Global state dictionary to pass client data (avoids context.state serialization issues)
_client_state = {}


def simulation(trainloaders, valloaders, testloader, device, dataset, num_classes, model, **cfg):
    global _client_state
    
    # Get strategy name and configuration
    strategy_name = cfg.get("aggregation", "fedavg")
    strategy_cfg = cfg.get("strategy_config", {})
    model_config = cfg.get("model_config", {})
    model_name = model_config.get("model")
    save_path = os.path.join(cfg.get("save_path", ""), cfg.get("experiment_name", ""), "models")
    os.makedirs(save_path, exist_ok=True)

    # Get strategy factory function using getattr on aggregation module
    strategy_func = getattr(agg, f"get_{strategy_name}", None)
    
    if strategy_func is None:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Supported: fedprox, fedavg, fedadagrad, fedadam, fedyogi, krum, dp_fedavg_adaptive, qfedavg, faulttolerant_fedavg.")
    
    strategy = strategy_func(save_path, num_classes, testloader, device, model_name, model_config, model, **strategy_cfg)
    
    # Store client data in global state (avoids serialization issues with context.state)
    _client_state = {
        "device": device,
        "dataset": dataset,
        "num_classes": num_classes,
        "trainloaders": trainloaders,
        "valloaders": valloaders,
        "mu": cfg.get("mu", 0.1),
        "model_config": model_config,
    }

    # Wrapper to inject state from global scope before calling client_fn
    def client_fn_wrapper(context: Context):
        # Inject global state into context for client_fn to access
        context._user_state = _client_state
        return original_client_fn(context)

    return fl.simulation.start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=cfg.get("num_clients", 7),
        config=fl.server.ServerConfig(num_rounds=cfg.get("rounds", 22)),
        strategy=strategy,
        client_resources=cfg.get("resources", {"num_cpus": 1, "num_gpus": 0.25})
    )
