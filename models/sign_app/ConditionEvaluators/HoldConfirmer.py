"""Confirma una letra/número cuando la misma predicción se mantiene estable por un tiempo."""

import time
from typing import Optional

from .InputBehavior import InputBehavior
from ..TextLogic import TextAction


class HoldConfirmer(InputBehavior):
    """
    Confirma una letra/número cuando la misma predicción se mantiene
    estable por `hold_seconds`. Para repetir la misma letra dos veces
    seguidas (ej. "SS") hay que soltar la mano o cambiar de seña
    brevemente entre una confirmación y la siguiente.
    """

    color = (0, 220, 0)
    label = "sostener"

    def __init__(self, hold_seconds: float):
        self.hold_seconds = hold_seconds
        self._stable_pred = None
        self._stable_since = None
        self._last_confirmed = None

    def update(self, hand_present, current_pred, text_so_far) -> Optional[TextAction.TextAction]:
        now = time.time()

        if current_pred != self._stable_pred:
            self._stable_pred = current_pred
            self._stable_since = now
            if current_pred is None:
                self._last_confirmed = None  # soltó la mano -> permite repetir letra
            return None

        if current_pred is None:
            return None

        held_for = now - self._stable_since
        if held_for >= self.hold_seconds and current_pred != self._last_confirmed:
            self._last_confirmed = current_pred
            return TextAction.TextAction(text_delta=current_pred)

        return None

    def progress(self) -> float:
        if self._stable_pred is None or self._stable_since is None:
            return 0.0
        elapsed = time.time() - self._stable_since
        return min(elapsed / self.hold_seconds, 1.0)