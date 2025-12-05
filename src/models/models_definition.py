from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)
import torch.nn as nn
import sys


def build_custom_layer4_fc_resnet18(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Congela tudo
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Descongela só o final da rede
    for name, param in model.layer4.named_parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

    return model


def build_simple_tl_resnet18(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for name, param in model.named_parameters():
        param.requires_grad = False   # <-- congela TUDO

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_custom_fc_resnet18(num_classes):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False  # <-- congela TUDO

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, in_features//2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features//2, num_classes)
    )

    return model


def build_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Congela tudo exceto classifier
    for name, param in model.named_parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    return model


def build_mobilenet_v3_small(num_classes):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    # congela tudo exceto classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_fn = getattr(sys.modules[__name__], f"build_{model_name}", None)
    if model_fn is None:
        raise ValueError(
        f"Unknown model '{model_name}'. Available builders: "
        f"{[name.replace('build_', '') for name in globals() if name.startswith('build_')] }"
        )
    
    return model_fn(num_classes)
