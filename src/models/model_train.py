import copy
import torch
from torch import nn
import numpy as np
from typing import List
from collections import OrderedDict
from .optimizers import get_optimizer_class, build_scheduler
from .criterion import build_criterion
from .metrics import compute_metrics


def train(model,
          dataset,
          trainloader,
          valloader,
          device,
          model_config,
          num_classes,
          global_params=None,
          mu=0.01,
          aggregation_method=None,
          verbose=False
):

    epochs = model_config['epochs']
    criterion = model_config['criterion']
    optimizer_class = model_config['optimizer']
    lr = model_config['learning_rate']
    lr_scheduler_config = model_config.get('lr_scheduler', None)
    patience = model_config.get('patience', 3)
    weight_decay = model_config.get('weight_decay', 0.0)

    criterion = build_criterion(criterion, dataset=dataset, device=device)

    # Weight decay + only trainable params
    optimizer_class = get_optimizer_class()[optimizer_class]
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Scheduler para estabilizar FL
    scheduler = build_scheduler(optimizer, lr_scheduler_config)

    model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(epochs):

        model.train()
        correct, total = 0, 0
        epoch_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if aggregation_method == "FedProx" and global_params is not None:
                prox = 0.0
                for w, w_global in zip(model.parameters(), global_params):
                    prox += ((w - w_global.to(device)) ** 2).sum()
                loss = loss + (mu / 2) * prox

            loss.backward()

            # Clipping para estabilizar FL
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

        # Final do epoch
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total if total > 0 else 0.0

        # Step do scheduler
        scheduler.step()

        # Avaliação no cliente
        val_metrics, _ = evaluate(model, dataset, valloader, device, num_classes=num_classes, model_config=model_config)
        val_loss = val_metrics['loss']

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, acc {epoch_acc:.4f}, val_loss {val_loss:.4f}")

    # Retorna o melhor estado
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


def evaluate(model, dataset, testloader, device, num_classes, model_config):
    model.to(device)
    criterion = build_criterion(model_config['criterion'], dataset=dataset, device=device)
    model.eval()

    total, correct = 0, 0
    running_loss = 0.0

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # loss ponderado pelo tamanho do batch
            running_loss += loss.item() * labels.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    # loss médio real
    loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob, axis=0)

    return compute_metrics(loss, y_true, y_pred, y_prob, num_classes)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]):
    state_dict_keys = list(model.state_dict().keys())
    new_state = OrderedDict()
    for k, arr in zip(state_dict_keys, parameters):
        ref = model.state_dict()[k]
        t = torch.tensor(arr, dtype=ref.dtype, device=ref.device)
        new_state[k] = t
    model.load_state_dict(new_state, strict=True)
