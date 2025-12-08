import flwr as fl
from flwr.common import Context
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
        """Evaluate model on validation set and return metrics for aggregation.
        
        Returns:
            tuple: (loss, num_examples, metrics_dict)
                - loss: float, validation loss value
                - num_examples: int, number of validation examples
                - metrics_dict: dict, all validation metrics with float values
        """
        self.set_parameters(parameters)
        self.model.to(self.device)

        # Evaluate on client validation set
        metrics, _ = model_train.evaluate(
            self.model,
            self.dataset,
            self.valloader,
            self.device,
            num_classes=self.num_classes,
            model_config=self.model_config
        )
        
        # Ensure all metrics are floats for proper aggregation
        metrics_clean = {k: float(v) if not isinstance(v, float) else v 
                        for k, v in metrics.items()}
        
        return float(metrics_clean["loss"]), len(self.valloader.dataset), metrics_clean


def client_fn(context: Context):
    """Factory do cliente FedProx - Flower Context API."""
    # Extract client ID from node_config
    cid = str(context.node_config.get("node_id", "0"))
    
    # Get client data from context's user state (injected by simulation wrapper)
    user_state = getattr(context, "_user_state", {})
    device = user_state.get("device")
    dataset = user_state.get("dataset")
    num_classes = user_state.get("num_classes")
    trainloaders = user_state.get("trainloaders")
    valloaders = user_state.get("valloaders")
    mu = user_state.get("mu", 0.1)
    model_config = user_state.get("model_config", {})
    
    model_name = model_config["model"]
    model = models_definition.build_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    return KidneyClient(
        dataset=dataset,
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        num_classes=num_classes,
        mu=mu,
        model_config=model_config
    )
