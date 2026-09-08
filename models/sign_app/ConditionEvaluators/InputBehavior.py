"""
Interfaz común para los comportamientos automáticos de entrada de
texto (confirmar letra, agregar espacio, y lo que se agregue después).

Para agregar un comportamiento nuevo: crear una clase que herede de
InputBehavior en su propio archivo, implementar `update()` (y
opcionalmente `progress()`), y agregar UNA instancia a la lista
`behaviors` en realtime_infer.py -- ni el loop principal ni
OverlayRenderer necesitan tocarse (Open/Closed).
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..TextLogic import TextAction


class InputBehavior(ABC):
    color: tuple[int, int, int] = (200, 200, 200)  # color de su barra de progreso
    label: str = "behavior"

    @abstractmethod
    def update(self, hand_present: bool, current_pred: Optional[str], text_so_far: str) -> Optional[TextAction.TextAction]:
        """Se llama una vez por frame. Devuelve una TextAction si corresponde
        modificar el texto este frame, o None si no corresponde nada."""
        raise NotImplementedError

    def progress(self) -> float:
        """Fracción [0, 1] para la barra de progreso visual (0 si no aplica)."""
        return 0.0