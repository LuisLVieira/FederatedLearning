import os
from torch import save
from ..models import model_train, models_definition

def server_evaluate_fn(num_classes, testloader, device, save_path: str = '.'):
    def evaluate_fn(server_round: int, parameters, config):
        model = models_definition.build_model(num_classes)

        model_train.set_parameters(model, parameters)
        metrics, cm = model_train.evaluate(model, testloader, device, num_classes)
        loss = float(metrics['loss'])

        os.makedirs(save_path, exist_ok=True)

        if server_round != 0:
            save(model.state_dict(), f"{save_path}/model_round{server_round}.pt")

        return loss, metrics

    return evaluate_fn