import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import KidneyData
import torchvision.transforms as transforms
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# Balanced Virtual Dataset (NO SAVING — ALL TRANSFORMS ON FETCH)
# -------------------------------------------------------------
class HeterogeneousDataHandler(Dataset):
    """
    Creates a virtual larger balanced dataset for class imbalance.
    Augmentation is applied only when __getitem__ is called.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset
        labels = base_dataset.metadata['target'].values

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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        return self.base[base_idx]


def load_datasets(
    clients: dict,
    test_df: pd.DataFrame,
    transforms_config: dict,
    batch_size: int = 32,
    label_map=dict,
    heterogeneous_data_handler=False
):
    img_size = transforms_config["img_size"]

    # -------------------------------
    # CT-SAFE medical image transforms
    # -------------------------------
    if heterogeneous_data_handler:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(7, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                fill=0
            ),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),

            transforms.RandomHorizontalFlip(p=0.1),

            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    trainloaders, valloaders = [], []

    for client_id, data in clients.items():

        train_dataset = KidneyData(data["train"], label_map, transform)
        val_dataset = KidneyData(data["valid"], label_map, transform)

        # Apply virtual expansion only when requested
        if heterogeneous_data_handler:
            train_dataset = HeterogeneousDataHandler(train_dataset)
            val_dataset = HeterogeneousDataHandler(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)

    test_dataset = KidneyData(test_df, label_map, transform)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloaders, valloaders, testloader

def plot_load_data(trainloaders, valloaders, num_clients, label_map):
    reverse_map = {v: k for k, v in label_map.items()}
    fig, axes = plt.subplots(2, num_clients, figsize=(22, 8), sharey=True)
    axes = axes.flatten()

    labels = {}
    for i, (trainloader, valloader) in enumerate(zip(trainloaders, valloaders)):
        print(f'client_{i+1}')

        train_labels = list(
            itertools.chain.from_iterable(labels.numpy() for _, labels in trainloader)
        )
        val_labels = list(
            itertools.chain.from_iterable(labels.numpy() for _, labels in valloader)
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


    plt.tight_layout()
    plt.show()
