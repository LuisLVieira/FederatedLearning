import os
from functools import partial
from .aggregation import get_fedprox, get_fedavg
from .clients import client_fn

os.environ["FLWR_SIMULATION_USE_RAY"] = "0"
os.environ["FLWR_LOGGING"] = "error"

import flwr as fl


def simulation(**simulation_cfg):
    # Allow caller to provide a ready-made strategy instance via `strategy` key.
    strategy = simulation_cfg.get("strategy", None)
    
    # Get required parameters from config
    device = simulation_cfg.get("device")
    num_classes = simulation_cfg.get("num_classes")
    model_name = simulation_cfg.get("model_name")
    trainloaders = simulation_cfg.get("trainloaders")
    valloaders = simulation_cfg.get("valloaders")
    
    # Create a partial function with the required parameters
    client_fn_with_params = partial(
        client_fn,
        device=device,
        num_classes=num_classes,
        model_name=model_name,
        trainloaders=trainloaders,
        valloaders=valloaders
    )

    return fl.simulation.start_simulation(
        client_fn=client_fn_with_params,
        num_clients=simulation_cfg.get("num_clients", 7),
        config=fl.server.ServerConfig(num_rounds=simulation_cfg.get("rounds", 22)),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
    )