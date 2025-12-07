import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import timm
from torchvision import transforms
from pycaret.classification import *
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_PATH = "./data/kidneyData.csv"     # ajuste se necessário
IMG_ROOT = "./data"                    # root onde as imagens estão
OUTPUT_DIR = "./results/pycaret"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKBONES = [
    "resnet18",
    "mobilenetv3_small_100",
    "efficientnet_b0",
]


# ============================================================
# CARREGAR DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

# Se o path no CSV for relativo, montar corretamente
df["full_path"] = df["path"].apply(lambda p: os.path.join(IMG_ROOT, p))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(model_name):
    print(f"\nExtraindo embeddings com backbone: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
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


# ============================================================
# PYCARET BENCHMARK FOR EACH BACKBONE
# ============================================================

results_summary = []

for backbone in BACKBONES:
    # 1) Extrair embeddings
    emb_df = extract_features(backbone)

    # 2) Rodar PyCaret para comparar modelos
    print(f"\nRodando PyCaret para backbone: {backbone}")
    s = setup(
        data=emb_df,
        target="target",
        session_id=42,
        silent=True,
        verbose=False,
        normalize=True,
    )

    best_model = compare_models(sort="Accuracy")
    table = pull()
    table["backbone"] = backbone

    # salvar tabela individual
    table.to_csv(f"{OUTPUT_DIR}/results_{backbone}.csv", index=False)

    # registrar melhor modelo
    best_row = table.iloc[0]
    results_summary.append(best_row)

    print(f"Melhor modelo para {backbone}:")
    print(best_row)


# ============================================================
# SALVAR RESUMO FINAL
# ============================================================

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(f"{OUTPUT_DIR}/summary_all_backbones.csv", index=False)

print("\n=====================================================")
print("FINALIZADO!")
print(f"Arquivos salvos em: {OUTPUT_DIR}")
print("=====================================================")
