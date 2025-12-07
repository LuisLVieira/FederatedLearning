import os
import json
import optuna
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import torch
import timm
from torchvision import transforms


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================

CSV_PATH = "/home/adrianovss/MO839A/data/kidneyData.csv"
IMG_ROOT = "/home/adrianovss/MO839A/data"
OUTPUT_DIR = "./results/optuna"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKBONES = [
    "resnet18",
    "mobilenetv3_small_100",
    "efficientnet_b0",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# CARREGAR CSV
# ===============================================================

df = pd.read_csv(CSV_PATH)
df["full_path"] = df["path"].apply(
    lambda p: os.path.join("/home/adrianovss/MO839A", p.lstrip("./"))
)


# ===============================================================
# FUNÇÃO DE EXTRAÇÃO DE EMBEDDINGS
# ===============================================================

def extract_features(model_name):
    print(f"\n🔍 Extraindo embeddings com backbone: {model_name}")

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    feats = []
    for img_path in tqdm(df["full_path"], desc=f"Extract {model_name}"):
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(img).cpu().numpy()[0]

        feats.append(emb)

    emb_df = pd.DataFrame(feats)
    emb_df["target"] = df["target"].values
    return emb_df


# ===============================================================
# OTIMIZAÇÃO COM OPTUNA PARA ESCOLHER O MELHOR MODELO
# ===============================================================

def objective(trial, X, y):

    model_type = trial.suggest_categorical("model", ["logreg", "rf", "svm"])

    if model_type == "logreg":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        model = LogisticRegression(C=C, max_iter=500)

    elif model_type == "rf":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 25)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )

    elif model_type == "svm":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        model = SVC(C=C, gamma=gamma)

    # pipeline com normalização
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    scores = cross_val_score(pipe, X, y, scoring="accuracy", cv=3)
    return scores.mean()


# ===============================================================
# LOOP PRINCIPAL DO EXPERIMENTO
# ===============================================================

final_results = []

for backbone in BACKBONES:

    emb_df = extract_features(backbone)

    X = emb_df.drop("target", axis=1).values
    y = emb_df["target"].values

    print(f"Rodando Optuna para {backbone}")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=25, show_progress_bar=True)

    best_params = study.best_params
    print(f"Melhor configuração {backbone}: {best_params}")

    # Treinar modelo final com os melhores hiperparâmetros
    model_type = best_params["model"]

    if model_type == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=best_params["C"], max_iter=500)),
        ])

    elif model_type == "rf":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"]
            )),
        ])

    else:  # svm
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=best_params["C"], gamma=best_params["gamma"])),
        ])

    # Avaliação final train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = {
        "backbone": backbone,
        "model_type": model_type,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="weighted")),
        "recall": float(recall_score(y_test, preds, average="weighted")),
        "precision": float(precision_score(y_test, preds, average="weighted")),
        "best_params": best_params,
    }

    final_results.append(result)

    # salvar resultados individuais
    with open(f"{OUTPUT_DIR}/{backbone}_best.json", "w") as f:
        json.dump(result, f, indent=4)

# salvar resumo geral
pd.DataFrame(final_results).to_csv(f"{OUTPUT_DIR}/summary_all_backbones.csv", index=False)

print("\n=====================================================")
print("FINALIZADO COM SUCESSO")
print(f"Resultados salvos em: {OUTPUT_DIR}")
print("=====================================================")
