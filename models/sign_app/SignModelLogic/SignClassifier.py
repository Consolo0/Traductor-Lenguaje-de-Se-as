"""
Clasificador de señas: envuelve el modelo entrenado (MLP) más el
enmascarado por modo (alfabético/numérico).

Punto de extensión: para probar otra arquitectura, heredar de
SignClassifier y sobreescribir `_build_model`.
"""

import json
from pathlib import Path

import torch

from .SignMLP import SignMLP


class SignClassifier:
    def __init__(self, model_path: Path, classes_path: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(classes_path) as f:
            self.classes: list[str] = json.load(f)

        self.model = self._build_model(len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self._mode_masks = {
            "numeric": [i for i, c in enumerate(self.classes) if c.isdigit()],
            "alpha": [i for i, c in enumerate(self.classes) if c.isalpha()],
        }

    def _build_model(self, n_classes: int) -> torch.nn.Module:
        return SignMLP(input_dim=63, n_classes=n_classes)

    def predict(self, features: list[float], mode: str) -> tuple[str, float]:
        """
        Devuelve (letra_mas_probable, confianza) restringido a las clases
        del modo activo. Quien llama decide qué umbral de confianza aceptar.
        """
        x = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)[0]

        allowed_idx = self._mode_masks[mode]
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits[allowed_idx] = logits[allowed_idx]

        probs = torch.softmax(masked_logits, dim=0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        return self.classes[pred_idx], confidence