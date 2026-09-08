"""Acción que un InputBehavior puede pedir sobre el texto acumulado."""

from dataclasses import dataclass


@dataclass
class TextAction:
    """Texto a agregar al acumulado (una letra, un espacio, etc.)."""
    text_delta: str