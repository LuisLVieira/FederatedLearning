import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

results = '/mnt/d/Mestrado/MO839/FederatedLearning/results'

epochs = glob.glob(f'{results}/*epoch*/global_val_metrics*csv')
rounds = glob.glob(f'{results}/*round*/global_val_metrics*csv')
models = glob.glob(f'{results}/fedavg*/global_val_metrics*csv')
aggregations = glob.glob(f'{results}/*KidneyData/global_val_metrics*csv')

agg_dict = {}
for agg in aggregations:
    df = pd.read_csv(agg)
    agg_dict[f'{os.path.basename((os.path.dirname(agg)).split("_")[0])}'] = df.at[0, 'f1-score']

epoch_dict = {}
for epoch in epochs:
    df = pd.read_csv(epoch)
    epoch_dict[f'{(os.path.dirname(epoch).split("_")[-1])}'] = df.at[0, 'f1-score']

model_dict = {}
for model in models:
    if 'KidneyData' in model:
        continue
    df = pd.read_csv(model)
    model_dict[f'{(os.path.dirname(model).split("_")[1])}'] = df.at[0, 'f1-score']

round_dict = {}
for round in rounds:
    df = pd.read_csv(round)
    round_dict[f'{(os.path.dirname(round).split("_")[-1])}'] = df.at[0, 'f1-score']

print(agg_dict)
print(epoch_dict)
print(model_dict)
print(round_dict)

translate = {
    "customFCResNet18": "ResNet18 \n(FC custom)",
    "customLayer4FCResNet18": "ResNet18 \n(Layer4 + FC)",
    "EfficientNetB0": "EfficientNet B0",
    "MobileNet": "MobileNet V2",
    "ResNet18": "ResNet18"
}

model_dict = {translate[k]: v for k, v in model_dict.items()}

agg_translate = {
    "faulttolerant": "Fault-Tolerant\n FedAvg",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedyogi": "FedYogi",
    "qfedavg": "q-FedAvg",
    "krum": "Krum"
}

agg_dict = {agg_translate[k]: v for k, v in agg_dict.items()}

# scientific font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

plt.suptitle('Federated Learning Variations', fontsize=18, fontweight="bold", y=0.98)

plt.subplots_adjust(wspace=0.25, hspace=0.35)

# ================= Subplot 1 =================
axes[0, 0].bar(agg_dict.keys(), agg_dict.values())
axes[0, 0].set_xlabel("Strategies", fontsize=12)
axes[0, 0].set_ylabel("F1-score", fontsize=12)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0, 0].tick_params(axis='x', rotation=25)
axes[0, 0].set_title("(a) Strategies", loc='left', fontsize=13)

# ================= Subplot 2 =================
axes[0, 1].bar(epoch_dict.keys(), epoch_dict.values())
axes[0, 1].set_xlabel("Local Epochs", fontsize=12)
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.4)
axes[0, 1].set_title("(b) Local Epochs", loc='left', fontsize=13)

# ================= Subplot 3 =================
axes[1, 0].bar(model_dict.keys(), model_dict.values())
axes[1, 0].set_xlabel("Models", fontsize=12)
axes[1, 0].set_ylabel("F1-score", fontsize=12)
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.4)
axes[1, 0].tick_params(axis='x', rotation=25)
axes[1, 0].set_title("(c) Models", loc='left', fontsize=13)

# ================= Subplot 4 =================
axes[1, 1].bar(round_dict.keys(), round_dict.values())
axes[1, 1].set_xlabel("Training Rounds", fontsize=12)
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.4)
axes[1, 1].set_title("(d) Training Rounds", loc='left', fontsize=13)

# remove top/right spines (clean style)
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(os.path.join(results, 'variations.png'), dpi=300, bbox_inches='tight')