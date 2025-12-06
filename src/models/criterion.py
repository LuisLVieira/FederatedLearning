from sklearn.utils.class_weight import compute_class_weight
from torch import nn, tensor, float
import numpy as np


def build_criterion(criterion_config, dataset=None, device=None):
    name = next(iter(criterion_config.keys()))
    params = criterion_config[name]


    if name == "FocalLoss" and dataset is not None and device is not None:
        name = "CrossEntropyLoss"
        class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(dataset["Class"]),
        y=dataset["Class"]
    )
        class_weights = tensor(class_weights, dtype=float).to(device)
        params['weight'] = class_weights
    
    LossClass = getattr(nn, name)

    return LossClass(**params)
