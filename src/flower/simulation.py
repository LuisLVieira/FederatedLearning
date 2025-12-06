import os
from functools import partial
import aggregation as agg
from .clients import client_fn

os.environ["FLWR_SIMULATION_USE_RAY"] = "0"
os.environ["FLWR_LOGGING"] = "error"

import flwr as fl


def simulation(trainloaders, valloaders, testloader, device, num_classes, **cfg):
    # Get strategy name and configuration
    strategy_name = cfg.get("aggregation", "fedavg")
    strategy_cfg = cfg.get("strategy_config", {})
    model_config = cfg.get("model_config", {})
    model_name = model_config.get("model_name")
    save_path = os.path.join(cfg.get("save_path", ""), cfg.get("experiment_name", ""), "models")
    os.makedirs(save_path, exist_ok=True)


    # Get strategy factory function using getattr on aggregation module
    strategy_func = getattr(agg, f"get_{strategy_name}", None)
    
    if strategy_func is None:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Supported: fedprox, fedavg.")
    
    strategy = strategy_func(save_path, num_classes, testloader, device, model_name, model_config, **strategy_cfg)
    
    # Create a partial function with the required parameters
    client_fn_with_params = partial(
        client_fn,
        device=device,
        num_classes=num_classes,
        model_name=model_name,
        trainloaders=trainloaders,
        valloaders=valloaders,
        mu=cfg.get("mu", 0.1)
    )

    return fl.simulation.start_simulation(
        client_fn=client_fn_with_params,
        num_clients=cfg.get("num_clients", 7),
        config=fl.server.ServerConfig(num_rounds=cfg.get("rounds", 22)),
        strategy=strategy,
        client_resources=cfg.get("resources", {"num_cpus": 1, "num_gpus": 0.25})
    )