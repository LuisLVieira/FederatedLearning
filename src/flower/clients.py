import flwr as fl
from collections import OrderedDict
from torch import tensor
from models import model_train, models_definition

class KidneyClient(fl.client.NumPyClient):
    def __init__(
        self,
        dataset,
        model,
        trainloader,
        valloader,
        device,
        num_classes,
        mu,
        model_config,    # <<<<< ADICIONADO
    ):
        self.dataset = dataset
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_classes = num_classes
        self.mu = mu
        self.model_config = model_config   # <<<<< SALVO AQUI

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Recebe pesos globais
        self.set_parameters(parameters)
        self.model.to(self.device)

        # Guarda pesos globais para FedProx
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # Usa o model_config CORRETO
        trained_model = model_train.train(
            model=self.model,
            dataset=self.dataset,
            trainloader=self.trainloader,
            valloader=self.valloader,
            device=self.device,
            model_config=self.model_config,  # <<<<< AQUI ESTÁ A CORREÇÃO PRINCIPAL
            num_classes=self.num_classes,
            global_params=global_params,
            mu=self.mu,
            aggregation_method="FedProx",
            verbose=False
        )

        return (
            [val.cpu().numpy() for _, val in trained_model.state_dict().items()],
            len(self.trainloader.dataset),
            {}
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)

        # usar model_config correto
        metrics, _ = model_train.evaluate(
            self.model,
            self.dataset,
            self.valloader,
            self.device,
            num_classes=self.num_classes,
            model_config=self.model_config     # <<<<< TAMBÉM AQUI
        )
        return float(metrics["loss"]), len(self.valloader.dataset), metrics


def client_fn(cid: str, device, num_classes, trainloaders, valloaders, mu, model_config):
    """Factory do cliente FedProx."""
    model_name = model_config["model"]

    model = models_definition.build_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    return KidneyClient(
        dataset=None,
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        num_classes=num_classes,
        mu=mu,
        model_config=model_config
    )
