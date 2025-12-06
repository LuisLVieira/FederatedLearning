import flwr as fl
from collections import OrderedDict
from torch import tensor
from ..models import model_train, models_definition

class KidneyClient(fl.client.NumPyClient):
    def __init__(
        self,
        dataset
        model,
        trainloader,
        valloader,
        device,
        num_classes
):
        self.dataset = dataset
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_classes = num_classes

    # --------------------------------------------------------
    # RETORNA OS PARÂMETROS LOCAIS
    # --------------------------------------------------------
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # --------------------------------------------------------
    # RECEBE OS PESOS DO MODELO GLOBAL (VENDA DO SERVIDOR)
    # --------------------------------------------------------
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # --------------------------------------------------------
    # FIT — AGORA COM FEDPROX
    # --------------------------------------------------------
    def fit(self, parameters, config):
        # 1. RECEBE PESOS GLOBAIS
        self.set_parameters(parameters)
        self.model.to(self.device)

        # 2. SALVA PESOS GLOBAIS PARA O FEDPROX
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # 3. TREINAMENTO LOCAL
        trained_model = model_train.train(
            model=self.model,
            dataset=self.dataset,
            trainloader=self.trainloader,
            valloader=self.valloader,
            device=self.device,
            patience=2,
            global_params=global_params,   # <--
            mu=0.1,               # <--
            verbose=False
        )
        # 4. RETORNA OS PESOS LOCAIS APÓS O TREINO
        return (
            [val.cpu().numpy() for _, val in trained_model.state_dict().items()],
            len(self.trainloader.dataset),
            {}
        )

    # --------------------------------------------------------
    # EVALUATE (sem mudanças)
    # --------------------------------------------------------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)

        metrics, _ = model_train.evaluate(self.model, self.valloader, self.device, num_classes=self.num_classes)
        return float(metrics["loss"]), len(self.valloader.dataset), metrics


def client_fn(cid: str, device, num_classes, model_name, trainloaders, valloaders):
    """Create a Kidney client representing a single organization."""

    model = models_definition.build_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]


    return KidneyClient(model, trainloader, valloader, device, num_classes)
