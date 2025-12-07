import os
from torch import save
from models import model_train, models_definition

def server_evaluate_fn(num_classes, testloader, device, model_name, model_config, save_path: str = '.'):
    def evaluate_fn(server_round: int, parameters, config):
        model = models_definition.build_model(model_name=model_name, num_classes=num_classes)

        model_train.set_parameters(model, parameters)
        
        metrics, _ = model_train.evaluate(
            model=model,
            dataset=None,
            testloader=testloader,
            device=device,
            num_classes=num_classes,
            model_config=model_config
        )

        loss = float(metrics['loss'])

        os.makedirs(save_path, exist_ok=True)

        if server_round != 0:
            save(model.state_dict(), f"{save_path}/model_round{server_round}.pt")

        return loss, metrics

    return evaluate_fn