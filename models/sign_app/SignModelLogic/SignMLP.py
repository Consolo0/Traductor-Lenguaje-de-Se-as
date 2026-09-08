"""Arquitectura del MLP que clasifica la seña a partir de los 63 features de landmarks."""

import torch.nn as nn


class SignMLP(nn.Module):
    """
    MLP simple: 63 features de entrada (21 landmarks x,y,z) -> letra/número.

    Arquitectura chica a propósito: el problema es fácil una vez que ya
    tenemos landmarks limpios (no imágenes crudas), así que no hace
    falta una red grande.
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