import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch import tensor, long
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import log
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

class KidneyData(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        label_map: dict,
        img_column: str = "path",
        target_column: str = "Class",
        transform=None
):
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map
        self.img_column = img_column
        self.target_column = target_column

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row[self.img_column]
        label_str = str(row[self.target_column])
        label = int(self.label_map[label_str])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, tensor(label, dtype=long)
    

def plot_random_kidneydata(
        data : pd.DataFrame,
        label_map: dict,
        save_path: str,
        experiment_name: str,
        target_column: str = "Class"
):
    classes = data[target_column].unique()

    fig, ax = plt.subplots(1, len(classes), figsize=(12, 12))
    for i, c in enumerate(classes):
        class_df = data[data[target_column] == c]
        data = KidneyData(class_df, label_map=label_map)
        img, label = data.__getitem__(random.randint(1, len(class_df)))
        log.logger.info(f"image {c} format {np.array(img).shape}")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(c)
        plt.axis('off')

    if save_path:
        plt.savefig(os.path.join(save_path, experiment_name, 'data_samples.png'))


def split_on_train_test(
        data: pd.DataFrame,
        test_ratio: float,
        random_seed: int,
        group_column: str = "group",
        target_column: str = "Class"
):
    splitter = StratifiedGroupKFold(n_splits=int(1/test_ratio), shuffle=True, random_state=random_seed)
    clients_idx, test_idx = next(splitter.split(
        data,
        data[target_column],
        groups=data[group_column]
        )
    )

    clients_data = data.iloc[clients_idx].reset_index(drop=True)
    test_data  = data.iloc[test_idx].reset_index(drop=True)

    return clients_data, test_data

def show_train_test_info(
        train_data,
        test_data,
        save_path,
        experiment_name: str = "",
        plot=True,
        group_column: str = "group",
        target_column: str = "Class"
):
    log.logger.info(f"Clients subjects: {train_data[group_column].nunique()}")
    log.logger.info(f"Test subjects: {test_data[group_column].nunique()}")
    log.logger.info(f"Intersection: {set(train_data[group_column]) & set(test_data[group_column])}")
    if plot:
        plt.figure(figsize=(8, 4))

        sns.countplot(x=test_data[target_column])
        plt.title("Data available to server only for test model")

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, experiment_name, 'test_data_counts.png'))


def split_on_clients(
        clients_df: pd.DataFrame,
        num_clients: int = 10,
        val_frac: float = 0.15,
        random_seed = 42,
        group_column: str = "group",
        target_column: str = "Class"
    ):
    """
    Splits data in NUM_CLIENTS clients, randomly, splitting in train and validation data inside each client (like small datasets)

    """
    unique_groups = clients_df["group"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    group_splits = np.array_split(unique_groups, num_clients)

    clients = {}

    for i, split in enumerate(group_splits):
        client_df = clients_df[clients_df[group_column].isin(split)]
        splitter = GroupShuffleSplit(n_splits=1, test_size = val_frac, random_state=random_seed)
        train_idx, val_idx = next(splitter.split(client_df, client_df[target_column], groups=client_df[group_column]))

        client_train_df = client_df.iloc[train_idx].reset_index(drop=True)
        client_val_df = client_df.iloc[val_idx].reset_index(drop=True)

        clients[i] = {'train': client_train_df, 'valid': client_val_df}

    return clients


def split_on_non_iid_clients(
        clients_df: pd.DataFrame,
        num_clients: int = 10,
        val_frac: float = 0.15,
        random_seed = 42,
        alpha: float = 0.5,
        group_column: str = "group",
        target_column: str = "Class"
):
    """
    Non-IID usando distribuição Dirichlet.
    - Garante todas as classes em todos os clientes
    - Mantém heterogeneidade controlada
    - Evita clientes com classes faltando
    """
    
    np.random.seed(42)
    clients = {}

    # Converte labels para numpy
    labels = clients_df[target_column].values
    unique_classes = np.unique(labels)

    # Índices do dataframe completo
    all_indices = np.arange(len(clients_df))

    # Lista de índices para cada cliente
    client_indices = [[] for _ in range(num_clients)]

    # Para cada classe, divide seus índices entre os clientes
    for cls in unique_classes:
        cls_idx = all_indices[labels == cls]

        # distribuição Dirichlet (controla Non-IID)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (proportions * len(cls_idx)).astype(int)

        start = 0
        for cid, amount in enumerate(proportions):
            client_indices[cid].extend(cls_idx[start:start + amount])
            start += amount

    # Embaralha
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

        cid_df = clients_df.iloc[client_indices[cid]].reset_index(drop=True)

        # split train/validation por cliente
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_seed)
        train_idx, val_idx = next(splitter.split(cid_df, cid_df[target_column], groups=cid_df[group_column]))

        client_train_df = cid_df.iloc[train_idx].reset_index(drop=True)
        client_val_df = cid_df.iloc[val_idx].reset_index(drop=True)

        clients[cid] = {'train': client_train_df, 'valid': client_val_df}

    return clients


def plot_class_distribution_on_clients(
        clients,
        num_clients,
        save_path,
        figname,
        experiment_name: str = "",
        target_column="Class",
        plot=True
):
    fig, axes = plt.subplots(2, num_clients, figsize=(22, 8), sharey=True)
    axes = axes.flatten()

    for i, (client_id, data) in enumerate(clients.items()):
        # Plot Train
        sns.countplot(x=data["train"][target_column], ax=axes[i])
        axes[i].set_title(f"Client {client_id} - Train")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

        # Plot Validation
        sns.countplot(x=data["valid"]["Class"], ax=axes[i + num_clients])
        axes[i + num_clients].set_title(f"Client {client_id} - Val")
        axes[i + num_clients].set_xlabel("")
        axes[i + num_clients].set_ylabel("")

    plt.tight_layout()
    plt.suptitle("Class Distribution per Client", fontsize=16)
    plt.subplots_adjust(top=0.88)
    if plot:
        plt.show()

    if save_path:
        plt.savefig(os.path.join(save_path, experiment_name,f'{figname}.png'))