"""
Entrena un MLP sobre los landmarks extraídos (data/landmarks/landmarks.csv)
para clasificar la letra/número de la seña.

Uso:
    python train.py --input "data/landmarks/landmarks.csv" --output "models/mlp_model.pt"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class SignMLP(nn.Module):
    """
    MLP simple: 63 features de entrada (21 landmarks x,y,z) -> letra/número.

    Arquitectura chica a propósito: el problema es fácil una vez que ya
    tenemos landmarks limpios (no imágenes crudas), así que no hace falta
    una red grande -- una red grande acá arriesgaría overfitting sin
    ninguna ganancia real de capacidad.
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder


def train_model(model, X_train, y_train, X_val, y_val, epochs: int, lr: float, device: str):
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()
            print(f"Epoch {epoch:>4} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    return model


def print_top_confusions(y_test, preds, label_encoder, top_n: int = 15):
    """
    Muestra los pares (clase_real -> clase_predicha) que más se confunden
    entre sí, ordenados de mayor a menor cantidad de casos. Ignora la
    diagonal de la matriz de confusión (los aciertos).
    """
    labels = label_encoder.classes_
    cm = confusion_matrix(y_test, preds, labels=range(len(labels)))

    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], labels[i], labels[j]))

    confusions.sort(reverse=True)

    print(f"\n=== Top {top_n} confusiones (real -> predicho) ===")
    for count, real, pred in confusions[:top_n]:
        print(f"  {real} -> {pred}: {count} casos")


def evaluate(model, X_test, y_test, label_encoder, device: str):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    print(f"\n=== Accuracy en test set: {acc:.4f} ===\n")

    report = classification_report(
        y_test, preds, target_names=label_encoder.classes_, zero_division=0
    )
    print("=== Reporte por clase (precision / recall / f1) ===")
    print(report)

    print_top_confusions(y_test, preds, label_encoder)

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de landmarks (data/landmarks/landmarks.csv)")
    parser.add_argument("--output", default="models/mlp_model.pt", help="Ruta para guardar el modelo entrenado")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    X, y, label_encoder = load_data(Path(args.input))
    print(f"Dataset: {X.shape[0]} ejemplos, {X.shape[1]} features, {len(label_encoder.classes_)} clases")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = SignMLP(input_dim=X.shape[1], n_classes=len(label_encoder.classes_)).to(device)

    model = train_model(model, X_train, y_train, X_test, y_test, args.epochs, args.lr, device)

    evaluate(model, X_test, y_test, label_encoder, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    # Guardamos también las clases del label encoder -- realtime_infer.py
    # necesita saber a qué letra/número corresponde cada índice de salida.
    classes_path = output_path.with_suffix(".classes.json")
    with open(classes_path, "w") as f:
        json.dump(list(label_encoder.classes_), f)

    print(f"\nModelo guardado en: {output_path}")
    print(f"Clases guardadas en: {classes_path}")


if __name__ == "__main__":
    main()