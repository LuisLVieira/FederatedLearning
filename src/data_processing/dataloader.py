from unicodedata import name
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .dataset import KidneyData
import torchvision.transforms as transforms
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

class HeterogeneousDataHandler(Dataset):
    """
    Creates a virtual larger balanced dataset for class imbalance.
    Applies base transforms + additional augmentations on __getitem__.
    """

    def __init__(self, base_dataset, heterogeneous_augmentations=None):
        self.base = base_dataset
        labels = base_dataset.metadata["target"].values

        class_counts = np.bincount(labels)
        max_count = max(class_counts)

        self.indices = []

        for cls, count in enumerate(class_counts):
            cls_idx = np.where(labels == cls)[0]

            repeat_factor = max_count // count
            remainder = max_count % count

            # duplicate full sets
            for _ in range(repeat_factor):
                self.indices.extend(cls_idx)

            # add random extra samples
            if remainder > 0:
                extra = np.random.choice(cls_idx, remainder, replace=True)
                self.indices.extend(extra)

        np.random.shuffle(self.indices)
        
        # Additional augmentations to apply on top of base transforms
        self.heterogeneous_augmentations = transforms.Compose(heterogeneous_augmentations) if heterogeneous_augmentations else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base[base_idx]
        
        # Apply additional augmentations if configured
        if self.heterogeneous_augmentations:
            img = self.heterogeneous_augmentations(img)
        
        return img, label


def parse_augmentation(func_name, params):
    transform_map = {
        "Resize": transforms.Resize,
        "RandomRotation": transforms.RandomRotation,
        "RandomAffine": transforms.RandomAffine,
        "RandomApply": transforms.RandomApply,
        "GaussianBlur": transforms.GaussianBlur,
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    }

    transform_cls = transform_map[func_name]

    # Case 1: no nested transforms → pass params directly
    if not any(isinstance(v, dict) for v in params.values()):
        return transform_cls(**params)

    # Case 2: nested transforms detected
    inner_transforms = []
    other_params = {}

    for key, value in params.items():
        if isinstance(value, dict) and key in transform_map:
            # recursive build
            nested = parse_augmentation(key, value)
            inner_transforms.append(nested)
        else:
            # normal parameter
            other_params[key] = value

    # Some transforms expect the nested transforms as a list
    return transform_cls(inner_transforms, **other_params)


def parse_augmentations(augmentations_config):
    transforms_list = []
    for func_name, params in augmentations_config.items():
        transform = parse_augmentation(func_name, params)
        transforms_list.append(transform)
    return transforms_list


def data_transforms(transforms_config, augment=True):
    """
    Create transform pipelines for training or validation/test data.
    
    Args:
        transforms_config: Configuration dictionary
        augment: If True, applies augmentations (for training data).
                If False, only resize + normalize (for val/test data).
    """
    img_size = transforms_config["img_size"]
    transforms_list = [
        transforms.Resize((img_size, img_size))
    ]

    # Apply augmentations only if augment=True (training data)
    if augment and "augmentations" in transforms_config and transforms_config["augmentations"]:
        transforms_list.extend(parse_augmentations(transforms_config["augmentations"]))
    
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            transforms_config["normalizer"]["mean"],
            transforms_config["normalizer"]["std"]
        ),
    ])

    return transforms.Compose(transforms_list)


def get_heterogeneous_augmentations(transforms_config):
    """
    Get augmentations specific to heterogeneous data handler (balanced virtual dataset).
    These are applied ON TOP of the base transforms.
    """
    if (
        ("heterogeneous_data_handler" in transforms_config) and
        (transforms_config["heterogeneous_data_handler"]) and
        ("augmentations" in transforms_config["heterogeneous_data_handler"])
    ):
        return parse_augmentations(transforms_config["heterogeneous_data_handler"]["augmentations"])
    return []


def load_datasets(
    clients: dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transforms_config: dict,
    batch_size: int = 32,
    num_workers: int = 0,
    label_map=dict,
):
    # Training transforms: WITH augmentations
    train_transform = data_transforms(transforms_config, augment=True)
    
    # Validation/Test transforms: WITHOUT augmentations (only resize + normalize)
    val_test_transform = data_transforms(transforms_config, augment=False)
    
    # Get heterogeneous augmentations to apply on top for balanced virtual dataset
    heterogeneous_augmentations = get_heterogeneous_augmentations(transforms_config)

    trainloaders, valloaders = [], []

    for client_id, data in clients.items():

        # Training data: with augmentations
        train_dataset = KidneyData(data["train"], label_map, transform=train_transform)
        
        # Validation data: WITHOUT augmentations
        val_dataset = KidneyData(data["valid"], label_map, transform=val_test_transform)

        # Wrap with HeterogeneousDataHandler if configured
        if (
            "heterogeneous_data_handler" in transforms_config and
            transforms_config["heterogeneous_data_handler"]
        ):
            train_dataset = HeterogeneousDataHandler(train_dataset, heterogeneous_augmentations)
            val_dataset = HeterogeneousDataHandler(val_dataset, heterogeneous_augmentations)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)

    # Test transforms: WITHOUT augmentations
    test_dataset = KidneyData(test_df, label_map, transform=val_test_transform)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test transforms: WITHOUT augmentations
    val_dataset = KidneyData(val_df, label_map, transform=val_test_transform)
    globalvalloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloaders, valloaders, globalvalloader, testloader


def plot_load_data(
        trainloaders,
        valloaders,
        num_clients,
        label_map,
        plot=True,
        save_path="",
        experiment_name="",
        figname="class_distribution"
    ):
    print('Plotting class distribution on loaded data...')
    reverse_map = {v: k for k, v in label_map.items()}
    fig, axes = plt.subplots(2, num_clients, figsize=(22, 8), sharey=True)
    axes = axes.flatten()

    labels = {}
    for i, (trainloader, valloader) in enumerate(zip(trainloaders, valloaders)):
        print(f'client_{i+1}')

        train_labels = list(
            itertools.chain.from_iterable(batch_labels.numpy() for _, batch_labels in trainloader)
        )
        val_labels = list(
            itertools.chain.from_iterable(batch_labels.numpy() for _, batch_labels in valloader)
        )

        train_mapped = [reverse_map[x] for x in train_labels]
        val_mapped   = [reverse_map[x] for x in val_labels]

        labels[i] = (train_mapped, train_mapped)

        ax_train = axes[i]
        sns.countplot(x=train_mapped, ax=ax_train)
        ax_train.set_title(f"Client {i+1} - Train")
        ax_train.set_xlabel("")
        ax_train.set_ylabel("")

        ax_val = axes[i + num_clients]
        sns.countplot(x=val_mapped, ax=ax_val)
        ax_val.set_title(f"Client {i+1} - Val")
        ax_val.set_xlabel("")
        ax_val.set_ylabel("")
    
    plt.suptitle("Class Distribution per Client after balancing", fontsize=16)
    plt.subplots_adjust(top=0.88)

    plt.tight_layout()
    if plot:
        plt.show()

    if save_path:
        plt.savefig(os.path.join(save_path, experiment_name, f'{figname}.png'))
