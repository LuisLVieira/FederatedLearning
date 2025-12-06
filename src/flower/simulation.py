import os
from .aggregation import get_fedprox, get_fedavg
from .clients import client_fn

os.environ["FLWR_SIMULATION_USE_RAY"] = "0"
os.environ["FLWR_LOGGING"] = "error"

import flwr as fl


def simulation(**simulation_cfg):
    # Allow caller to provide a ready-made strategy instance via `strategy` key.
    strategy = simulation_cfg.get("strategy", None)

    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=simulation_cfg.get("num_clients", 7),
        config=fl.server.ServerConfig(num_rounds=simulation_cfg.get("rounds", 22)),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
    )